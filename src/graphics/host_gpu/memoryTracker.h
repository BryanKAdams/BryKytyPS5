#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORYTRACKER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORYTRACKER_H_

#include "common/assert.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/regionManager.h"

#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

namespace Libs::Graphics {

class MemoryTracker final {
public:
	explicit MemoryTracker(PageManager& page_manager);
	~MemoryTracker();

	KYTY_CLASS_NO_COPY(MemoryTracker);

	[[nodiscard]] bool IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	// !IsRegionGpuModified && IsRegionCpuModified, taking each region's lock once.
	[[nodiscard]] bool IsRegionOnlyCpuModified(uint64_t vaddr, uint64_t size);
	void               MarkRegionAsCpuModified(uint64_t vaddr, uint64_t size);
	void               MarkRegionAsGpuModified(uint64_t vaddr, uint64_t size);
	void               UnmarkRegionAsGpuModified(uint64_t vaddr, uint64_t size);
	void               UntrackMemory(uint64_t vaddr, uint64_t size);

	// Asynchronous readback. A recorded download arms the GPU-dirty pages it copies with a unique
	// token and its publication tick; the publication callback finalizes (un-dirties) only pages
	// still armed with its token. Any GPU-dirty transition in between disarms the page, so a GPU
	// write recorded after the download is never lost. Armed pages stay GPU-dirty (NoAccess)
	// until finalized, and waiters use their tick instead of draining the GPU.
	[[nodiscard]] ReadbackState QueryReadback(uint64_t vaddr, uint64_t size);
	void ArmReadback(uint64_t vaddr, uint64_t size, uint64_t token, uint64_t tick);
	void FinalizeReadback(uint64_t vaddr, uint64_t size, uint64_t token);
	// Hot pages (guest threads keep reading them after GPU writes; see BufferCache) are the only
	// ones GrantStaleRead opens.
	void SetReadbackHot(uint64_t vaddr, uint64_t size, bool hot) {
		CheckNotInUploadCallback();
		Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			std::scoped_lock lock(manager->lock);
			manager->SetReadbackHot(manager->GetCpuAddr() + offset, bytes, hot);
		});
	}
	// Relaxed readback: when every GPU-dirty page of the range is armed and hot, lets guest
	// threads read those pages with their previous bytes until the download publishes them or
	// the GPU writes them again. Returns whether it did.
	[[nodiscard]] bool GrantStaleRead(uint64_t vaddr, uint64_t size) {
		CheckNotInUploadCallback();
		bool granted = true;
		Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			std::scoped_lock lock(manager->lock);
			granted = manager->GrantStaleRead(manager->GetCpuAddr() + offset, bytes) && granted;
		});
		return granted;
	}
	[[nodiscard]] bool HasArmedPages(uint64_t vaddr, uint64_t size);
	// Lock-free and possibly stale: whether the tracker page holding vaddr is GPU-dirty. Only
	// for choosing a read path whose outcome stays correct either way.
	[[nodiscard]] bool IsPageGpuDirtyHint(uint64_t vaddr) const noexcept {
		const auto index = vaddr / TRACKER_REGION_SIZE;
		if (index >= REGION_COUNT) {
			return false;
		}
		const auto* manager = m_regions[index].load(std::memory_order_acquire);
		return manager != nullptr && manager->GpuDirtyHint(vaddr);
	}

	// Lock-free: true when every tracker region the range touches exists and none of the 64 KiB
	// slices it touches holds a CPU-dirty page (see RegionManager::CpuDirtySlices). Then a locked
	// upload pass over the range would find nothing to copy and change nothing. False when unsure.
	[[nodiscard]] bool IsRegionCpuCleanHint(uint64_t vaddr, uint64_t size) const noexcept {
		if (size == 0 || vaddr >= TRACKER_ADDRESS_SIZE || size > TRACKER_ADDRESS_SIZE - vaddr) {
			return false;
		}
		constexpr uint64_t SLICE_SIZE = RegionManager::CPU_DIRTY_SLICE_PAGES * TRACKER_PAGE_SIZE;
		const uint64_t     end        = vaddr + size;
		for (uint64_t address = vaddr; address < end;) {
			const auto  index   = address / TRACKER_REGION_SIZE;
			const auto* manager = m_regions[index].load(std::memory_order_acquire);
			if (manager == nullptr) {
				return false;
			}
			const auto region_start = index * TRACKER_REGION_SIZE;
			const auto region_end   = region_start + TRACKER_REGION_SIZE;
			const auto first_slice  = (address - region_start) / SLICE_SIZE;
			const auto last_slice   = (std::min(end, region_end) - 1 - region_start) / SLICE_SIZE;
			const auto slices =
			    (last_slice == 63 ? ~uint64_t {0} : (uint64_t {1} << (last_slice + 1)) - 1) &
			    (~uint64_t {0} << first_slice);
			if ((manager->CpuDirtySlices() & slices) != 0) {
				return false;
			}
			address = region_end;
		}
		return true;
	}

	// One conservative hint per tracker region. CPU-dirty bits remain authoritative.
	// Dirty transitions, region creation, buffer registration and mapping publish hints.
	// A pass exchanges each word once before inspecting it; publications after that exchange
	// remain pending. Unfinished regions must be restored, never cleared a second time.
	// Summary words hold one bit per hint word, so a pass with nothing dirty reads 64 words
	// instead of 4096. A pass exchanges a summary word before the hint words it flags.
	static constexpr size_t      BDA_HINT_WORDS = TRACKER_ADDRESS_SIZE / TRACKER_REGION_SIZE / 64;
	static constexpr size_t      BDA_SUMMARY_WORDS = BDA_HINT_WORDS / 64;
	void                         PublishBdaHints(uint64_t vaddr, uint64_t size) noexcept;
	[[nodiscard]] uint64_t       ConsumeBdaSummaryWord(size_t summary) noexcept;
	void                         RestoreBdaSummary(size_t summary, uint64_t words) noexcept;
	[[nodiscard]] uint64_t       ConsumeBdaHintWord(size_t word) noexcept;
	void                         RestoreBdaHints(size_t word, uint64_t bits) noexcept;
	// Pending means the next pass will find the region: both its hint and summary bits are set.
	[[nodiscard]] bool           IsBdaHintPending(uint64_t region) const noexcept;
	[[nodiscard]] RegionManager* FindRegion(uint64_t region) const noexcept {
		EXIT_IF(region >= REGION_COUNT);
		return m_regions[region].load(std::memory_order_acquire);
	}
	[[nodiscard]] RegionBits SnapshotCpuDirty(RegionManager& manager);
	// Diagnostic used after a completed selective pass over mapped, registered owners.
	[[nodiscard]] bool BdaHintsCoverCpuDirty(uint64_t vaddr, uint64_t size);

	// Removes protection from a range and flushes GPU-owned data when required.
	template <typename Flush>
	void InvalidateRegion(uint64_t vaddr, uint64_t size, Flush&& on_flush) noexcept {
		static_assert(std::is_invocable_v<Flush&>);
		CheckNotInUploadCallback();

		Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			const bool should_flush = [&] {
				// Perform both the GPU modification check and CPU state change with the lock in
				// case the GPU thread is racing to mark the page modified. If a flush is needed,
				// on_flush performs the CPU state change.
				std::scoped_lock lock(manager->lock);
				if (manager->IsModified<DirtySource::Gpu>(offset, bytes)) {
					return true;
				}
				manager->ChangeState<DirtySource::Cpu, true>(manager->GetCpuAddr() + offset, bytes);
				return false;
			}();
			if (should_flush) {
				on_flush();
			}
		});
	}
#if KYTY_BUILD == KYTY_BUILD_DEBUG
	void ValidateGpuDirtyPages(const RangeSet& dirty, uint64_t vaddr, uint64_t size,
	                           const char* operation) const noexcept;
	void ValidateGpuDirtyOwnership(const RangeSet& dirty, uint64_t vaddr, uint64_t size,
	                               const char* operation);
#else
	void ValidateGpuDirtyPages(const RangeSet&, uint64_t, uint64_t, const char*) const noexcept {}
	void ValidateGpuDirtyOwnership(const RangeSet&, uint64_t, uint64_t, const char*) {}
#endif

	// Runs of GPU-dirty pages whose bytes no recorded download covers yet.
	template <typename Func>
	void ForEachUnarmedDownloadRange(uint64_t vaddr, uint64_t size, Func&& func) {
		static_assert(std::is_nothrow_invocable_v<Func&, uint64_t, uint64_t>);
		CheckNotInUploadCallback();
		Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			std::scoped_lock lock(manager->lock);
			manager->ForEachUnarmedGpuRange(manager->GetCpuAddr() + offset, bytes, func);
		});
	}

	template <bool clear, typename Func>
	void ForEachDownloadRange(uint64_t vaddr, uint64_t size, Func&& func) {
		static_assert(std::is_nothrow_invocable_v<Func&, uint64_t, uint64_t>);
		CheckNotInUploadCallback();
		Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			std::scoped_lock lock(manager->lock);
			const auto       address = manager->GetCpuAddr() + offset;
			manager->template ForEachModifiedRange<DirtySource::Gpu, false>(address, bytes, func);
			if constexpr (clear) {
				manager->template ChangeState<DirtySource::Gpu, false>(address, bytes);
			}
		});
	}

	template <typename RangeFunc, typename UploadFunc>
	void ForEachUploadRange(uint64_t vaddr, uint64_t size, bool is_written, RangeFunc&& range_func,
	                        UploadFunc&& upload_func) {
		static_assert(std::is_nothrow_invocable_v<RangeFunc&, uint64_t, uint64_t>);
		static_assert(std::is_nothrow_invocable_v<UploadFunc&>);
		CheckNotInUploadCallback();
		Iterate<true>(vaddr, size, [](RegionManager*, uint64_t, uint64_t) {});
		const auto* previous_upload_owner = std::exchange(s_upload_owner, this);
		Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			manager->lock.lock();
			manager->ForEachModifiedRange<DirtySource::Cpu, true>(manager->GetCpuAddr() + offset,
			                                                      bytes, range_func);
			if (!is_written) {
				manager->lock.unlock();
			}
		});
		upload_func();
		if (is_written) {
			Iterate<false>(vaddr, size,
			               [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
				               manager->template ChangeState<DirtySource::Gpu, true>(
				                   manager->GetCpuAddr() + offset, bytes);
				               manager->lock.unlock();
			               });
		}
		s_upload_owner = previous_upload_owner;
	}

private:
	static constexpr size_t REGION_COUNT = TRACKER_ADDRESS_SIZE / TRACKER_REGION_SIZE;
	inline static thread_local const MemoryTracker* s_upload_owner = nullptr;

	void CheckNotInUploadCallback() const noexcept {
		if (s_upload_owner == this) {
			EXIT("memory tracker re-entered from upload callback\n");
		}
	}

	template <bool create, typename Func>
	bool Iterate(uint64_t vaddr, uint64_t size, Func&& func) {
		ValidateRange(vaddr, size);
		using Result = std::invoke_result_t<Func, RegionManager*, uint64_t, uint64_t>;
		constexpr bool returns_bool = std::is_same_v<Result, bool>;
		uint64_t       remaining    = size;
		uint64_t       index        = vaddr / TRACKER_REGION_SIZE;
		uint64_t       offset       = vaddr % TRACKER_REGION_SIZE;
		while (remaining != 0) {
			const auto bytes   = std::min(TRACKER_REGION_SIZE - offset, remaining);
			auto*      manager = m_regions[index].load(std::memory_order_acquire);
			if (manager == nullptr && create) {
				manager = GetOrCreateRegion(index);
			}
			if (manager != nullptr) {
				if constexpr (returns_bool) {
					if (func(manager, offset, bytes)) {
						return true;
					}
				} else {
					func(manager, offset, bytes);
				}
			}
			remaining -= bytes;
			offset = 0;
			index++;
		}
		return false;
	}

	static void    ValidateRange(uint64_t vaddr, uint64_t size);
	RegionManager* GetOrCreateRegion(uint64_t index);

	std::unique_ptr<std::atomic<RegionManager*>[]> m_regions;
	std::vector<std::unique_ptr<RegionManager>>    m_region_storage;
	std::mutex                                     m_region_mutex;
	PageManager&                                   m_page_manager;
	std::unique_ptr<std::atomic<uint64_t>[]>       m_bda_hints;
	std::unique_ptr<std::atomic<uint64_t>[]>       m_bda_summary;
	// Pages armed by recorded downloads across all regions; zero lets queries skip the walk.
	std::atomic<int64_t>                           m_armed_pages {0};
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORYTRACKER_H_
