#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/faultManager.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <array>
#include <chrono>
#include <deque>
#include <map>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;
class TextureCache;

using BufferId = Common::SlotId;
inline constexpr BufferId NULL_BUFFER_ID {0};

class BufferCache {
public:
	static constexpr uint32_t CACHING_PAGEBITS  = 14;
	static constexpr uint64_t CACHING_PAGESIZE  = uint64_t {1} << CACHING_PAGEBITS;
	static constexpr uint64_t CACHING_NUMPAGES  = uint64_t {1} << (40 - CACHING_PAGEBITS);
	static constexpr uint64_t BDA_PAGETABLE_SIZE =
	    CACHING_NUMPAGES * sizeof(vk::DeviceAddress);

	BufferCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	            TextureCache& texture_cache);
	~BufferCache();
	KYTY_CLASS_NO_COPY(BufferCache);

	void                   InvalidateMemory(uint64_t vaddr, uint64_t size);
	void                   ReadMemory(uint64_t vaddr, uint64_t size, bool is_write = false);
	[[nodiscard]] Buffer&  GetBuffer(BufferId id) { return m_slot_buffers[id]; }
	[[nodiscard]] BufferId FindBuffer(uint64_t vaddr, uint64_t size);
	// FindBuffer, keeping `id` while it is a live buffer covering the range (as ObtainBuffer does):
	// buffers never overlap, so a live buffer over the range is the one FindBuffer would find.
	[[nodiscard]] BufferId RefindBuffer(BufferId id, uint64_t vaddr, uint64_t size) {
		return !IsBufferInvalid(id) && m_slot_buffers[id].IsInBounds(vaddr, size)
		           ? id
		           : FindBuffer(vaddr, size);
	}
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBuffer(uint64_t vaddr, uint64_t size,
	                                                        bool     is_written,
	                                                        bool     is_texel_buffer = false,
	                                                        BufferId id              = {});
	[[nodiscard]] StreamBuffer&                GetUtilityBuffer(MemoryUsage usage) noexcept {
		switch (usage) {
			case MemoryUsage::Upload: return m_staging_buffer;
			case MemoryUsage::Stream: return ActiveStream();
			case MemoryUsage::Download: return m_download_buffer;
			case MemoryUsage::DeviceLocal: return m_device_buffer;
		}
		EXIT("BufferCache: invalid utility-buffer usage\n");
	}
	// The ring that per-draw stream copies go to (see bufferCache.cpp).
	[[nodiscard]] StreamBuffer& ActiveStream() noexcept;
	// Device-addressable ring for per-draw parameter records that shaders load by address.
	[[nodiscard]] StreamBuffer& GetDrawRecordBuffer() noexcept { return m_draw_record_buffer; }
	[[nodiscard]] const Buffer* GetGdsBuffer() const noexcept { return &m_gds_buffer; }
	[[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept { return &m_bda_pagetable_buffer; }
	[[nodiscard]] Buffer* GetFaultBuffer() noexcept { return m_fault_manager.GetFaultBuffer(); }
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferForImage(uint64_t vaddr, uint64_t size);
	void FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds);
	void CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
	                bool src_gds);
	// Cache-index and exact dirty-range queries require GPU-thread serialization.
	[[nodiscard]] bool IsRegionRegistered(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool HasGpuDirtyBytes(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsPageGpuDirtyHint(uint64_t vaddr) const noexcept {
		return m_memory_tracker.IsPageGpuDirtyHint(vaddr);
	}
	// Changes whenever bytes become GPU-dirty or a download starts, which is also the only way a
	// GPU-dirty range can become clean again (see m_clean_pages). GPU thread only.
	[[nodiscard]] uint64_t GpuDirtyGeneration() const noexcept { return m_gpu_dirty_generation; }
	// Eager readback of hot pages: memory that CPU reads have faulted on. A write recorded to a
	// hot page is downloaded at the next flush point, so its bytes are usually published before
	// the CPU reads them again. OnCommandRecorded() marks writes of the bindings obtained so far
	// as recorded; only recorded writes are downloaded.
	void               OnCommandRecorded();
	void               RecordEagerReadbacks();
	void               ProcessFaultBuffer();
	void               SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size);
	void               PublishBdaHints(uint64_t vaddr, uint64_t size) noexcept;
	void               SynchronizeBdaLegacy(const RangeSet& mapped);
	[[nodiscard]] bool SynchronizeBdaSelective(const RangeSet& mapped);
	[[nodiscard]] bool CheckBdaHintInvariant(const RangeSet& mapped);
	void               RunGarbageCollector();

private:
	friend struct BufferCacheTestAccess;
	friend struct PerformanceMemoryTestAccess;

	bool IsBufferInvalid(BufferId id) const {
		const auto* buffer = m_slot_buffers.try_get(id);
		return buffer == nullptr || buffer->is_deleted;
	}

	using BufferMap = std::map<uint64_t, BufferId>;
	struct OverlapResult {
		BufferMap::iterator first;
		BufferMap::iterator last;
		uint64_t            begin;
		uint64_t            end;
		bool                has_stream_leap;
	};

	using PageTable = MultiLevelPageTable<BufferId, CACHING_PAGEBITS, 40, 16>;
	static_assert(CACHING_PAGESIZE == (uint64_t {1} << PageTable::kPageBits));
	void WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source, uint64_t size);
	void TouchBuffer(const Buffer& buffer);
	[[nodiscard]] OverlapResult ResolveOverlaps(uint64_t vaddr, uint64_t size);
	void JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score);
	[[nodiscard]] BufferId CreateBuffer(uint64_t vaddr, uint64_t size);
	void                   Register(BufferId id);
	void Unregister(BufferId id);
	template <bool insert>
	void ChangeRegister(BufferId id);
	void DeleteBuffer(BufferId id);
	[[nodiscard]] bool SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                     bool is_written, bool is_texel_buffer);
	[[nodiscard]] vk::Buffer UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
	                                      uint64_t total_size);
	[[nodiscard]] bool SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size);
	// Records downloads of the range's unarmed GPU-dirty pages, arms them, and queues their
	// publication, which finalizes them. Returns false when there was nothing to arm.
	[[nodiscard]] bool DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size);
	// Thread_Gpu: arms the readback window around [vaddr, vaddr+size).
	void RecordReadback(uint64_t vaddr, uint64_t size, bool is_write);
	// Other threads: publish GPU-dirty pages without draining Thread_Gpu.
	void ReadMemoryAsync(uint64_t vaddr, uint64_t size, bool is_write);
	void MarkReadbackHot(uint64_t vaddr);
	void QueueEagerReadback(uint64_t vaddr, uint64_t size);
	// Whether a recorded download of these bytes has not finished publishing.
	[[nodiscard]] bool InFlightIntersects(uint64_t vaddr, uint64_t size);
	void               PruneInFlight();
	[[nodiscard]] bool SynchronizeBdaWord(size_t word, const RangeSet& mapped);
	[[nodiscard]] bool SynchronizeBdaRegion(uint64_t region, const RangeSet& mapped);
	[[nodiscard]] bool SynchronizeDirtyOwners(const RegionBits& dirty, uint64_t region_begin,
	                                          uint64_t begin, uint64_t end);

	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	FaultManager                                      m_fault_manager;
	Buffer                                            m_gds_buffer;
	Buffer                                            m_bda_pagetable_buffer;
	Common::SlotVector<Buffer>                        m_slot_buffers;
	Common::LeastRecentlyUsedCache<BufferId, uint64_t> m_lru_cache;
	BufferMap                                         m_buffers;
	PageTable                                         m_page_table;
	RangeSet                                          m_gpu_modified_ranges;
	MemoryTracker                                     m_memory_tracker;
	StreamBuffer                                      m_staging_buffer;
	StreamBuffer                                      m_stream_buffer;
	StreamBuffer                                      m_download_buffer;
	StreamBuffer                                      m_device_buffer;
	StreamBuffer                                      m_draw_record_buffer;
	// KYTY_DEBUG_AB=streamhost: a stream ring in device memory for the A/B (see ActiveStream).
	std::unique_ptr<StreamBuffer> m_stream_device;
	TextureCache&                                     m_texture_cache;
	uint64_t                                          m_total_used_memory  = 0;
	uint64_t m_trigger_gc_memory  = 1ull * 1024 * 1024 * 1024;
	uint64_t m_critical_gc_memory = 2ull * 1024 * 1024 * 1024;
	uint64_t m_gc_tick            = 0;
	uint64_t m_readback_token     = 0;
	// Tracker pages hit by CPU read faults, with the tick of their latest fault.
	static constexpr size_t HotReadbackPages = 64;
	std::unordered_map<uint64_t, uint64_t> m_hot_pages;
	std::vector<uint64_t>                  m_eager_pending; // Written by unrecorded commands.
	std::vector<uint64_t>                  m_eager_ready;   // Written by recorded commands.
	// Downloaded byte ranges whose backing is not written yet, oldest first. They left
	// m_gpu_modified_ranges when recorded; the other bytes of their pages are already valid.
	struct InFlightDownload {
		uint64_t                                   tick = 0;
		std::vector<std::pair<uint64_t, uint64_t>> ranges;
	};
	std::deque<InFlightDownload> m_inflight_downloads;
	// HasGpuDirtyBytes: tracker pages found clean, valid while m_gpu_dirty_generation is
	// unchanged. It is bumped when bytes become GPU-dirty or a download starts; nothing else can
	// dirty a page. GPU thread only, like HasGpuDirtyBytes.
	struct CleanPage {
		uint64_t page       = UINT64_MAX;
		uint64_t generation = 0;
	};
	std::array<CleanPage, 64> m_clean_pages {};
	uint64_t                  m_gpu_dirty_generation = 1;
	// ObtainBuffer's stream-ring copies of the current BDA epoch, by range (see there).
	struct StreamCopy {
		uint64_t vaddr        = 0;
		uint64_t size         = 0;
		uint64_t epoch        = 0; // 0: none.
		uint64_t tick         = 0;
		uint64_t gpu_writes   = 0; // m_gpu_dirty_generation.
		uint64_t image_writes = 0; // TextureCache::GpuModifiedGeneration.
		Buffer*  stream       = nullptr;
		uint64_t offset       = 0;
	};
	static constexpr size_t StreamCopySlots = 4096;
	std::vector<StreamCopy> m_stream_copies = std::vector<StreamCopy>(StreamCopySlots);
	// KYTY_VERIFY_STREAM_REUSE=1: whether to copy again anyway (after comparing; see there).
	[[nodiscard]] static bool VerifyStreamReuse(const StreamCopy& copy);
	// ObtainBufferForImage's staged uploads, by range, with the time of the latest.
	struct ImageStage {
		uint64_t                              vaddr = 0;
		uint64_t                              size  = 0;
		std::chrono::steady_clock::time_point time {};
	};
	std::array<ImageStage, 256> m_image_stages {};
	[[nodiscard]] bool          IsRepeatedImageStage(uint64_t vaddr, uint64_t size);
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
