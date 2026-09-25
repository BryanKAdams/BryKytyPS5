#include "graphics/host_gpu/renderer/cache/bufferCache.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/drainStats.h"
#include "graphics/host_gpu/renderer/gpuZones.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "kernel/memory.h"

#include <algorithm>
#include <bit>
#include <cinttypes>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

namespace Libs::Graphics {

namespace {

constexpr uint64_t MiB           = 1024 * 1024;
constexpr uint64_t GdsBufferSize = 64 * 1024;

} // namespace

void BufferCache::WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source,
                                  uint64_t size) {
	auto* bytes = static_cast<const uint8_t*>(source);
	while (size != 0) {
		const auto chunk  = std::min(size, m_staging_buffer.Size());
		const auto offset = m_staging_buffer.Copy(bytes, chunk, 4);
		buffer.CopyFrom(m_scheduler.Current(), m_staging_buffer, offset, buffer.Offset(address),
		                chunk, vk::AccessFlagBits::eHostWrite);
		bytes += chunk;
		address += chunk;
		size -= chunk;
	}
}

void BufferCache::Register(BufferId id) {
	ChangeRegister<true>(id);
}

void BufferCache::Unregister(BufferId id) {
	ChangeRegister<false>(id);
}

template <bool insert>
void BufferCache::ChangeRegister(BufferId id) {
	auto& buffer = m_slot_buffers[id];
	PageTable::PageRange pages {};
	EXIT_IF(!PageTable::TryGetPageRange(buffer.CpuAddress(), buffer.Size(), pages));
	for (size_t page = pages.first; page < pages.last_exclusive; ++page) {
		if constexpr (insert) {
			m_page_table[page] = id;
		} else {
			m_page_table[page] = {};
		}
	}
	const auto size_pages = pages.last_exclusive - pages.first;
	if constexpr (insert) {
		const auto [it, inserted] = m_buffers.emplace(buffer.CpuAddress(), id);
		(void)it;
		EXIT_IF(!inserted);
		m_total_used_memory += buffer.Size();
		buffer.lru_id = m_lru_cache.Insert(id, m_gc_tick);
		std::vector<vk::DeviceAddress> addresses;
		addresses.reserve(size_pages);
		for (uint64_t i = 0; i < size_pages; ++i) {
			addresses.push_back(buffer.BufferDeviceAddress() + (i << CACHING_PAGEBITS));
		}
		WriteDataBuffer(m_bda_pagetable_buffer, pages.first * sizeof(vk::DeviceAddress),
		                addresses.data(), addresses.size() * sizeof(vk::DeviceAddress));
		// Publish the full resulting owner, including pages newly reachable after a merge.
		m_memory_tracker.PublishBdaHints(buffer.CpuAddress(), buffer.Size());
	} else {
		const auto found = m_buffers.find(buffer.CpuAddress());
		EXIT_IF(found == m_buffers.end() || found->second != id);
		m_buffers.erase(found);
		EXIT_IF(buffer.Size() > m_total_used_memory);
		m_total_used_memory -= buffer.Size();
		m_lru_cache.Free(buffer.lru_id);
		m_bda_pagetable_buffer.Fill(pages.first * sizeof(vk::DeviceAddress),
		                            size_pages * sizeof(vk::DeviceAddress), 0);
		buffer.is_deleted = true;
	}
}

void BufferCache::TouchBuffer(const Buffer& buffer) {
	if (!buffer.is_deleted) {
		m_lru_cache.Touch(buffer.lru_id, m_gc_tick);
	}
}

void BufferCache::DeleteBuffer(BufferId id) {
	if (IsBufferInvalid(id)) {
		return;
	}
	Unregister(id);
	if (m_scheduler.Active()) {
		m_scheduler.DeferOperation([this, id] { m_slot_buffers.erase(id); });
	} else {
		m_slot_buffers.erase(id);
	}
}

bool BufferCache::DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	std::vector<vk::BufferCopy>                copies;
	std::vector<std::pair<uint64_t, uint64_t>> pages;
	uint64_t                                   total_size     = 0;
	const auto                                 buffer_address = buffer.CpuAddress();
	// Pages armed by an earlier download already have their bytes in flight.
	m_memory_tracker.ForEachUnarmedDownloadRange(
	    vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
		    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, address, bytes,
		                                           "buffer download");
		    pages.emplace_back(address, bytes);
		    m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t end) {
			    copies.emplace_back(start - buffer_address, total_size, end - start);
			    // Keep packed ranges on separate cache lines, as in shadPS4.
			    total_size += Common::AlignUp(end - start, 64);
		    });
		    m_gpu_modified_ranges.Subtract(address, bytes);
	    });
	if (pages.empty()) {
		return false;
	}
	// Arm after any ring wrap (which can submit and move the tick) and before queueing the
	// publication, which may run inline during shutdown. Waiters only wait on submitted ticks,
	// so they cannot observe an arm before its publication is queued.
	const auto token = ++m_readback_token;
	const auto arm   = [this, &pages, token] {
		for (const auto& [address, bytes]: pages) {
			m_memory_tracker.ArmReadback(address, bytes, token, m_scheduler.CurrentTick());
		}
	};
	if (copies.empty()) {
		// Dirty pages without recorded bytes carry nothing to copy; their publication only
		// ends their GPU ownership.
		arm();
		m_scheduler.DeferPriorityOperation([this, token, pages = std::move(pages)] {
			for (const auto& [address, bytes]: pages) {
				m_memory_tracker.FinalizeReadback(address, bytes, token);
			}
		});
		return true;
	}
	DrainStats::Record(DrainStats::Kind::Readback, total_size);

	auto [mapped, offset] = m_download_buffer.Map(total_size, 64);
	std::unique_ptr<Buffer> temporary;
	Buffer*                 download = &m_download_buffer;
	if (mapped == nullptr) {
		// Ring wraps already wait for retirement. A reservation larger than the ring needs
		// separate host staging, kept alive until the priority callback publishes the bytes.
		temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                     vk::BufferUsageFlagBits::eTransferDst, total_size);
		download  = temporary.get();
		mapped    = download->Mapped().data();
		offset    = 0;
		EXIT_IF(mapped == nullptr);
	} else {
		m_download_buffer.Commit();
	}
	for (auto& copy: copies) {
		copy.dstOffset += offset;
	}

	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto              native = command.Handle();
	GpuZones::Mark(native, DrainStats::Zone::BufferCopy);
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = buffer.Handle();
	before.offset              = 0;
	before.size                = buffer.Size();
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before, 0,
	                       nullptr);
	native.copyBuffer(buffer.Handle(), download->Handle(), static_cast<uint32_t>(copies.size()),
	                  copies.data());

	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	after.buffer        = download->Handle();
	after.offset        = offset;
	after.size          = total_size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands |
	                           vk::PipelineStageFlagBits::eHost,
	                       {}, 0, nullptr, 1, &after, 0, nullptr);
	arm();
	PruneInFlight();
	auto& inflight = m_inflight_downloads.emplace_back();
	inflight.tick  = m_scheduler.CurrentTick();
	inflight.ranges.reserve(copies.size());
	for (const auto& copy: copies) {
		inflight.ranges.emplace_back(buffer_address + copy.srcOffset, copy.size);
	}
	m_scheduler.DeferPriorityOperation([this, token, pages = std::move(pages), download,
	                                    owner = std::move(temporary), mapped, offset, total_size,
	                                    buffer_address, copies = std::move(copies)] {
		download->Invalidate(offset, total_size);
		for (const auto& copy: copies) {
			Libs::LibKernel::Memory::WriteBacking(buffer_address + copy.srcOffset,
			                                      mapped + (copy.dstOffset - offset), copy.size);
		}
		(void)owner; // Retain dedicated staging through the final backing write.
		// The backing holds the GPU bytes now; pages still armed by this download become
		// accessible. A page re-dirtied meanwhile keeps its protection for the next download.
		for (const auto& [address, bytes]: pages) {
			m_memory_tracker.FinalizeReadback(address, bytes, token);
		}
	});
	return true;
}

BufferCache::BufferCache(GraphicContext& graphics, CommandScheduler& scheduler,
                         PageManager& page_manager, TextureCache& texture_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_fault_manager(graphics, scheduler, *this),
      m_gds_buffer(graphics, scheduler, MemoryUsage::Stream, 0, AllFlags, GdsBufferSize),
      m_bda_pagetable_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                             BDA_PAGETABLE_SIZE),
      m_memory_tracker(page_manager),
      m_staging_buffer(graphics, scheduler, MemoryUsage::Upload, 512 * MiB),
      m_stream_buffer(graphics, scheduler, MemoryUsage::Stream, 64 * MiB),
      m_download_buffer(graphics, scheduler, MemoryUsage::Download, 64 * MiB),
      m_device_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 128 * MiB),
      m_draw_record_buffer(graphics, scheduler, MemoryUsage::Stream, 4 * MiB,
                           vk::BufferUsageFlagBits::eShaderDeviceAddress),
      m_texture_cache(texture_cache) {
	std::memset(m_gds_buffer.Mapped().data(), 0, static_cast<size_t>(m_gds_buffer.Size()));
	m_gds_buffer.Flush(0, m_gds_buffer.Size());
	SetVulkanObjectNameF(m_graphics.device, m_bda_pagetable_buffer.Handle(),
	                     "BDA Page Table Buffer");
	const auto null_id =
	    m_slot_buffers.insert(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, 16);
	EXIT_IF(null_id != NULL_BUFFER_ID);
	SetVulkanObjectNameF(m_graphics.device, GetBuffer(null_id).Handle(), "Kyty.NullBuffer");
	if (!m_graphics.CanReportMemoryUsage()) {
		return;
	}
	constexpr int64_t GiB              = 1024ll * 1024 * 1024;
	constexpr int64_t target_threshold = 8 * GiB;
	const auto        budget =
	    static_cast<int64_t>(std::min<uint64_t>(m_graphics.GetTotalMemoryBudget(), INT64_MAX));
	const auto threshold = std::min(budget, target_threshold);
	const auto expected  = std::min(budget - 6 * threshold / 10, budget - GiB);
	const auto critical  = std::min(budget - 2 * threshold / 10, budget - GiB / 2);
	m_trigger_gc_memory  = static_cast<uint64_t>(std::max<int64_t>(expected, GiB));
	m_critical_gc_memory = static_cast<uint64_t>(std::max<int64_t>(critical, 2 * GiB));
}

BufferCache::~BufferCache() {
	if (!m_gpu_modified_ranges.Empty()) {
		EXIT("BufferCache: destroyed with pending GPU-modified ranges\n");
	}
	for (const auto& [vaddr, id]: m_buffers) {
		(void)vaddr;
		const auto& buffer = m_slot_buffers[id];
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: destroyed with GPU-modified buffer\n");
		}
	}
	m_buffers.clear();
}

void BufferCache::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid memory-invalidation range\n");
	}
	// A CPU write needs the whole range CPU-dirty. GPU-dirty pages are published first; a GPU
	// write recorded while this thread waits can re-dirty a page, so repeat until all are marked.
	for (;;) {
		bool gpu_dirty = false;
		m_memory_tracker.InvalidateRegion(vaddr, size, [&gpu_dirty] { gpu_dirty = true; });
		if (!gpu_dirty) {
			return;
		}
		ReadMemory(vaddr, size, true);
	}
}

void BufferCache::RecordReadback(uint64_t vaddr, uint64_t size, bool is_write) {
	if (is_write && !IsRegionRegistered(vaddr, size)) {
		if (m_memory_tracker.IsRegionGpuModified(vaddr, size)) {
			EXIT("BufferCache: GPU-dirty memory has no buffer owner, addr=0x%016" PRIx64
			     " size=0x%016" PRIx64 "\n",
			     vaddr, size);
		}
		return;
	}
	auto& buffer = m_slot_buffers[FindBuffer(vaddr, size)];

	// Widen nearby CPU reads so they share one publication.
	constexpr uint64_t WindowSize   = 512 * 1024;
	const auto         buffer_begin = buffer.CpuAddress();
	const auto         buffer_end   = buffer_begin + buffer.Size();
	const auto window_begin = std::max(Common::AlignDown(vaddr, WindowSize), buffer_begin);
	const auto window_end = std::min(std::max(window_begin + WindowSize, vaddr + size), buffer_end);
	if (!DownloadBufferMemory(buffer, window_begin, window_end - window_begin)) {
		DrainStats::Record(DrainStats::Kind::ReadbackClean, 0);
	}
}

void BufferCache::ReadMemory(uint64_t vaddr, uint64_t size, bool is_write) {
	if (!GuestGpu::IsGpuThread() && CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported buffer readback from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	if (!GuestGpu::IsGpuThread()) {
		ReadMemoryAsync(vaddr, size, is_write);
		return;
	}
	// Thread_Gpu consumes the bytes itself, so it waits here. Pages armed by an earlier
	// (already submitted) download only need their own tick, which is not a full drain.
	// Its reads do not mark pages hot: they read shader code sharing a page with GPU data, or
	// arguments written just before, where an eager download only delays the next wait.
	RecordReadback(vaddr, size, is_write);
	const auto state = m_memory_tracker.QueryReadback(vaddr, size);
	if (state.gpu_dirty) {
		EXIT_IF(state.unarmed);
		m_scheduler.Wait(state.tick);
		m_scheduler.WaitPriorityOperations(state.tick);
	}
	if (is_write) {
		m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
	}
}

void BufferCache::ReadMemoryAsync(uint64_t vaddr, uint64_t size, bool is_write) {
	// Thread_Gpu only records and submits the download; this thread waits for its own
	// publication while Thread_Gpu keeps working. The caller retries the access afterwards:
	// a page the GPU re-dirtied meanwhile simply faults again.
	auto     state = m_memory_tracker.QueryReadback(vaddr, size);
	uint64_t tick  = state.gpu_dirty ? state.tick : 0;
	if (!state.gpu_dirty) {
		return;
	}
	if (state.unarmed || tick >= m_scheduler.CurrentTick()) {
		const auto reason = DrainStats::CurrentReason();
		m_scheduler.Context().GetGpu().SendCommandSync([this, vaddr, size, is_write, reason, &tick] {
			DrainStats::ReasonScope reason_scope(reason);
			if (!is_write) {
				MarkReadbackHot(vaddr);
			}
			RecordReadback(vaddr, size, is_write);
			const auto current = m_memory_tracker.QueryReadback(vaddr, size);
			if (!current.gpu_dirty) {
				tick = 0;
				return;
			}
			EXIT_IF(current.unarmed);
			tick = current.tick;
			// A foreign thread may only wait on submitted ticks: nothing else is guaranteed
			// to submit the open command buffer while guest work waits on this thread.
			if (tick >= m_scheduler.CurrentTick()) {
				m_scheduler.Flush();
			}
		});
	}
	if (tick == 0) {
		return;
	}
	{
		DrainStats::WaitTimer wait(DrainStats::Kind::TickWait);
		m_scheduler.GetMasterSemaphore().Wait(tick);
	}
	m_scheduler.WaitPriorityOperations(tick);
}

void BufferCache::MarkReadbackHot(uint64_t vaddr) {
	const auto page = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto tick = m_scheduler.CurrentTick();
	if (auto it = m_hot_pages.find(page); it != m_hot_pages.end()) {
		it->second = tick;
		return;
	}
	if (m_hot_pages.size() >= HotReadbackPages) {
		// Replace the page whose latest fault is the oldest.
		m_hot_pages.erase(std::min_element(
		    m_hot_pages.begin(), m_hot_pages.end(),
		    [](const auto& a, const auto& b) { return a.second < b.second; }));
	}
	m_hot_pages.emplace(page, tick);
}

void BufferCache::QueueEagerReadback(uint64_t vaddr, uint64_t size) {
	const auto end = vaddr + size;
	for (const auto& [page, tick]: m_hot_pages) {
		(void)tick;
		if (page < end && page + TRACKER_PAGE_SIZE > vaddr) {
			m_eager_pending.push_back(page);
		}
	}
}

void BufferCache::OnCommandRecorded() {
	if (m_eager_pending.empty()) {
		return;
	}
	m_eager_ready.insert(m_eager_ready.end(), m_eager_pending.begin(), m_eager_pending.end());
	m_eager_pending.clear();
}

void BufferCache::RecordEagerReadbacks() {
	if (m_eager_ready.empty()) {
		return;
	}
	DrainStats::ReasonScope reason(DrainStats::Reason::EagerReadback);
	std::vector<uint64_t>   pages;
	pages.swap(m_eager_ready);
	std::sort(pages.begin(), pages.end());
	pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
	for (const auto page: pages) {
		// The written buffer may have been retired since; its pages then have no GPU owner.
		if (!IsRegionRegistered(page, TRACKER_PAGE_SIZE)) {
			continue;
		}
		// Buffers are CACHING_PAGESIZE-aligned, so one buffer owns the whole tracker page.
		auto& buffer = m_slot_buffers[FindBuffer(page, TRACKER_PAGE_SIZE)];
		(void)DownloadBufferMemory(buffer, page, TRACKER_PAGE_SIZE);
	}
}

BufferId BufferCache::FindBuffer(uint64_t vaddr, uint64_t size) {
	if (vaddr == 0) {
		return NULL_BUFFER_ID;
	}
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid buffer discovery request\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			return *owner;
		}
	}
	return CreateBuffer(vaddr, size);
}

BufferCache::OverlapResult BufferCache::ResolveOverlaps(uint64_t vaddr, uint64_t size) {
	static constexpr int      StreamLeapThreshold = 16;
	static constexpr uint64_t StreamLeapSize      = CACHING_PAGESIZE * 128;

	auto       begin      = vaddr;
	auto       end        = vaddr + size;
	const auto find_first = [&](uint64_t address) {
		auto first = m_buffers.lower_bound(address);
		if (first != m_buffers.begin()) {
			const auto  previous = std::prev(first);
			const auto& buffer   = m_slot_buffers[previous->second];
			if (buffer.CpuAddress() + buffer.Size() > address) {
				first = previous;
			}
		}
		return first;
	};
	auto first           = find_first(begin);
	auto last            = first;
	int  stream_score    = 0;
	bool has_stream_leap = false;
	for (; last != m_buffers.end() && last->first < end; ++last) {
		const auto& buffer        = m_slot_buffers[last->second];
		const auto  buffer_begin  = buffer.CpuAddress();
		const auto  buffer_end    = buffer_begin + buffer.Size();
		const bool  expands_left  = buffer_begin < begin;
		const bool  expands_right = buffer_end > end;
		begin                     = std::min(begin, buffer_begin);
		end                       = std::max(end, buffer_end);
		if (!has_stream_leap && (stream_score += buffer.StreamScore()) > StreamLeapThreshold) {
			has_stream_leap = true;
			// Reserve space in the incoming stream's direction of growth.
			// The old buffer extending left of the request predicts growth to the right, and vice versa.
			if (expands_left) {
				end += std::min(StreamLeapSize, PageTable::kAddressSpaceSize - end);
			}
			if (expands_right) {
				const auto minimum = CACHING_PAGESIZE * 2;
				if (begin > minimum) {
					begin -= std::min(StreamLeapSize, begin - minimum);
				}
				first = find_first(begin);
				begin = std::min(begin, first->first);
			}
		}
	}
	return {first, last, begin, end, has_stream_leap};
}

void BufferCache::JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score) {
	auto& new_buffer = m_slot_buffers[new_id];
	auto& overlap    = m_slot_buffers[overlap_id];
	if (accumulate_stream_score) {
		new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
	}
	new_buffer.CopyFrom(m_scheduler.Current(), overlap, 0,
	                    overlap.CpuAddress() - new_buffer.CpuAddress(), overlap.Size());
	DeleteBuffer(overlap_id);
}

BufferId BufferCache::CreateBuffer(uint64_t vaddr, uint64_t size) {
	EXIT_IF(m_scheduler.Current().IsInvalid());
	const auto end = Common::AlignUp(vaddr + size, CACHING_PAGESIZE);
	vaddr = Common::AlignDown(vaddr, CACHING_PAGESIZE);
	size               = end - vaddr;
	const auto overlap = ResolveOverlaps(vaddr, size);

	const auto id = m_slot_buffers.insert(
	    m_graphics, m_scheduler, MemoryUsage::DeviceLocal, overlap.begin,
	    AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, overlap.end - overlap.begin);
	const auto& buffer = m_slot_buffers[id];
	SetVulkanObjectNameF(m_graphics.device, buffer.Handle(),
	                     "Kyty.GameBuffer[guest=0x{:016x} size=0x{:x}]", overlap.begin,
	                     overlap.end - overlap.begin);
	for (auto it = overlap.first; it != overlap.last;) {
		const auto old_id = (it++)->second;
		JoinOverlap(id, old_id, !overlap.has_stream_leap);
	}
	Register(id);
	return id;
}

bool BufferCache::SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size, bool is_written,
                                    bool is_texel_buffer) {
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size = 0;
	vk::Buffer                  source;
	m_memory_tracker.ForEachUploadRange(
	    vaddr, size, is_written,
	    [&](uint64_t address, uint64_t bytes) noexcept {
		    copies.emplace_back(total_size, buffer.Offset(address), bytes);
		    total_size += bytes;
	    },
	    [&]() noexcept { source = UploadCopies(buffer, copies, total_size); });
	if (source) {
		auto& command = m_scheduler.Current();
		command.EndRendering();
		const auto native = command.Handle();
		GpuZones::Mark(native, DrainStats::Zone::BufferCopy);
		vk::BufferMemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
		                       vk::AccessFlagBits::eTransferRead |
		                       vk::AccessFlagBits::eTransferWrite;
		before.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer              = buffer.Handle();
		before.offset              = 0;
		before.size                = buffer.Size();
		native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                       vk::PipelineStageFlagBits::eTransfer,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &before, 0, nullptr);
		native.copyBuffer(source, buffer.Handle(), static_cast<uint32_t>(copies.size()),
		                  copies.data());
		auto after          = before;
		after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
		native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                       vk::PipelineStageFlagBits::eAllCommands,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &after, 0, nullptr);
	}
	if (is_texel_buffer && !is_written) {
		return SynchronizeBufferFromImage(buffer, vaddr, size);
	}
	return false;
}

vk::Buffer BufferCache::UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                                     uint64_t total_size) {
	if (copies.empty()) {
		return nullptr;
	}

	auto [mapped, base_offset] = m_staging_buffer.Map(total_size, 4);
	if (mapped != nullptr) {
		for (auto& copy: copies) {
			const auto address = buffer.CpuAddress() + copy.dstOffset;
			std::memcpy(mapped + copy.srcOffset, reinterpret_cast<const void*>(address), copy.size);
			copy.srcOffset += base_offset;
		}
		m_staging_buffer.Commit();
		return m_staging_buffer.Handle();
	}

	auto temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Upload, 0,
	                                         vk::BufferUsageFlagBits::eTransferSrc, total_size);
	for (const auto& copy: copies) {
		const auto address = buffer.CpuAddress() + copy.dstOffset;
		std::memcpy(temporary->Mapped().data() + copy.srcOffset,
		            reinterpret_cast<const void*>(address), copy.size);
	}
	temporary->Flush(0, total_size);
	const auto handle = temporary->Handle();
	m_scheduler.DeferOperation([owner = std::move(temporary)]() mutable { owner.reset(); });
	return handle;
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBuffer(uint64_t vaddr, uint64_t size,
                                                       bool is_written, bool is_texel_buffer,
                                                       BufferId id) {
	auto& command = m_scheduler.Current();
	if (command.IsInvalid() || !GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: buffer request requires a recording command buffer\n");
	}

	if (!is_written && size <= CACHING_PAGESIZE &&
	    !m_memory_tracker.IsRegionGpuModified(vaddr, size) &&
	    m_memory_tracker.IsRegionCpuModified(vaddr, size)) {
		const auto alignment = std::max<uint64_t>(
		    m_graphics.physical_device_properties.limits.minUniformBufferOffsetAlignment, 1);
		auto [mapped, offset] = m_stream_buffer.Map(size, alignment, false);
		if (mapped != nullptr && Libs::LibKernel::Memory::TryReadBacking(vaddr, mapped, size)) {
			m_stream_buffer.Commit();
			return {&m_stream_buffer, offset};
		}
	}

	if (IsBufferInvalid(id) || !m_slot_buffers[id].IsInBounds(vaddr, size)) {
		id = FindBuffer(vaddr, size);
	}
	auto& buffer = m_slot_buffers[id];
	TouchBuffer(buffer);
	(void)SynchronizeBuffer(buffer, vaddr, size, is_written, is_texel_buffer);
	if (is_written) {
		m_gpu_modified_ranges.Add(vaddr, size);
		m_texture_cache.OnBufferGpuWrite(vaddr, size);
		if (!m_hot_pages.empty()) {
			QueueEagerReadback(vaddr, size);
		}
		if (DrainStats::Enabled()) {
			DrainStats::RecordGpuWrite(vaddr, size);
			if (m_texture_cache.IsKnownDccMetadata(vaddr, size)) {
				DrainStats::Record(DrainStats::Kind::DccMetaWrite, size);
			}
		}
	}
	return {&buffer, buffer.Offset(vaddr)};
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBufferForImage(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid image source\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			TouchBuffer(buffer);
			(void)SynchronizeBuffer(buffer, vaddr, size, false, false);
			return {&buffer, buffer.Offset(vaddr)};
		}
	}
	if (IsRegionGpuModified(vaddr, size)) {
		return ObtainBuffer(vaddr, size, false, false);
	}

	auto [staging, stage_offset] = m_staging_buffer.Map(size, 16);
	if (staging == nullptr || (!Libs::LibKernel::Memory::TryReadBacking(vaddr, staging, size) &&
	                           !Libs::LibKernel::Memory::TryReadPrtBacking(vaddr, staging, size))) {
		EXIT("BufferCache: failed to read mapped guest image backing\n");
	}
	m_staging_buffer.Commit();
	return {&m_staging_buffer, stage_offset};
}

void BufferCache::FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds) {
	if ((vaddr & 3u) != 0 || size == 0 || (size & 3u) != 0 || size > UINT64_MAX - vaddr) {
		EXIT("BufferCache: fill range must be dword aligned\n");
	}
	if (is_gds) {
		if (vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - vaddr) {
			EXIT("BufferCache: GDS fill range is out of bounds\n");
		}
		m_gds_buffer.Fill(vaddr, size, value);
		return;
	}
	if (vaddr == 0) {
		EXIT("BufferCache: invalid fill memory address\n");
	}
	(void)m_texture_cache.ClearMeta(vaddr);
	if (!IsRegionGpuModified(vaddr, size)) {
		// Access the guest mapping so write faults invalidate cached buffers and images.
		auto* destination = reinterpret_cast<uint32_t*>(vaddr);
		std::fill(destination, destination + size / sizeof(uint32_t), value);
		return;
	}

	m_texture_cache.InvalidateMemoryFromGPU(vaddr, size, TextureCache::GpuWriteSource::Fill);
	auto [dst, dst_offset] = ObtainBuffer(vaddr, size, true, true);
	dst->Fill(dst_offset, size, value);
	OnCommandRecorded();
}

void BufferCache::CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
                             bool src_gds) {
	const bool dst_memory = !dst_gds;
	const bool src_memory = !src_gds;
	if ((dst_memory && dst_vaddr == 0) || (src_memory && src_vaddr == 0) || size == 0 ||
	    ((dst_gds || src_gds) && ((dst_vaddr | src_vaddr | size) & 3u) != 0) ||
	    size > UINT64_MAX - dst_vaddr || size > UINT64_MAX - src_vaddr || (dst_gds && src_gds) ||
	    (dst_gds && (dst_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - dst_vaddr)) ||
	    (src_gds && (src_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - src_vaddr))) {
		EXIT("BufferCache: invalid copy range, src=0x%016" PRIx64 " dst=0x%016" PRIx64
		     " size=0x%016" PRIx64 " src_gds=%d dst_gds=%d\n",
		     src_vaddr, dst_vaddr, size, static_cast<int>(src_gds), static_cast<int>(dst_gds));
	}
	if (src_memory && dst_memory && !IsRegionGpuModified(dst_vaddr, size) &&
	    !IsRegionGpuModified(src_vaddr, size) && !m_texture_cache.FindImageFromRange(src_vaddr, size)) {
		std::memcpy(reinterpret_cast<void*>(dst_vaddr), reinterpret_cast<const void*>(src_vaddr),
		            size);
		return;
	}

	auto& command = m_scheduler.Current();
	if (dst_memory) {
		m_texture_cache.InvalidateMemoryFromGPU(dst_vaddr, size, TextureCache::GpuWriteSource::Copy);
	}
	const auto src_id      = src_memory ? FindBuffer(src_vaddr, size) : BufferId {};
	const auto dst_id      = dst_memory ? FindBuffer(dst_vaddr, size) : BufferId {};
	auto [src, src_offset] = src_memory ? ObtainBuffer(src_vaddr, size, false, true, src_id)
	                                    : std::pair {&m_gds_buffer, src_vaddr};
	auto [dst, dst_offset] = dst_memory ? ObtainBuffer(dst_vaddr, size, true, true, dst_id)
	                                    : std::pair {&m_gds_buffer, dst_vaddr};
	dst->CopyFrom(command, *src, src_offset, dst_offset, size);
	OnCommandRecorded();
}

bool BufferCache::IsRegionRegistered(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid registered-region query\n");
	}
	// Cached buffers are ordered and non-overlapping. The last buffer beginning before the query
	// end is therefore the only possible intersection.
	const auto candidate = m_buffers.lower_bound(vaddr + size);
	if (candidate == m_buffers.begin()) {
		return false;
	}
	const auto& [address, id] = *std::prev(candidate);
	return address + m_slot_buffers[id].Size() > vaddr;
}

bool BufferCache::IsRegionGpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionGpuModified(vaddr, size);
}

bool BufferCache::HasGpuDirtyBytes(uint64_t vaddr, uint64_t size) {
	// Downloaded bytes have left the byte set, but their backing is valid only once published.
	// Other bytes of an armed page are valid already: a window download around a CPU fault
	// arms whole pages, and Thread_Gpu reads game tables that share such pages every frame.
	return m_gpu_modified_ranges.Intersects(vaddr, size) || InFlightIntersects(vaddr, size);
}

void BufferCache::PruneInFlight() {
	while (!m_inflight_downloads.empty() &&
	       m_scheduler.IsPublished(m_inflight_downloads.front().tick)) {
		m_inflight_downloads.pop_front();
	}
}

bool BufferCache::InFlightIntersects(uint64_t vaddr, uint64_t size) {
	PruneInFlight();
	const auto end = vaddr + size;
	for (const auto& download: m_inflight_downloads) {
		for (const auto& [address, bytes]: download.ranges) {
			if (address < end && address + bytes > vaddr) {
				return true;
			}
		}
	}
	return false;
}

bool BufferCache::IsRegionCpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionCpuModified(vaddr, size);
}

void BufferCache::RunGarbageCollector() {
	const auto tick = m_gc_tick++;
	if (m_graphics.CanReportMemoryUsage()) {
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
	}
	if (m_total_used_memory < m_trigger_gc_memory) {
		return;
	}
	DrainStats::ReasonScope reason(DrainStats::Reason::BufferGc);

	const bool     aggressive = m_total_used_memory >= m_critical_gc_memory;
	const uint64_t age        = std::min<uint64_t>(aggressive ? 80 : 160, tick);
	const size_t   limit      = aggressive ? 64 : 32;

	std::vector<BufferId> dirty_buffers;
	size_t                retire_count = 0;
	m_lru_cache.ForEachItemBelow(tick - age, [&](BufferId id) {
		auto& buffer = m_slot_buffers[id];
		EXIT_IF(buffer.is_deleted);
		m_memory_tracker.ValidateGpuDirtyOwnership(m_gpu_modified_ranges, buffer.CpuAddress(),
		                                           buffer.Size(), "garbage collection");
		const bool dirty = m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size());
		if (dirty && !aggressive) {
			return false;
		}
		if (dirty) {
			// Pages armed by an in-flight readback publish with the wait below.
			EXIT_IF(!DownloadBufferMemory(buffer, buffer.CpuAddress(), buffer.Size()) &&
			        !m_memory_tracker.HasArmedPages(buffer.CpuAddress(), buffer.Size()));
			dirty_buffers.push_back(id);
		} else {
			m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
			DeleteBuffer(id);
		}
		return ++retire_count == limit;
	});
	if (dirty_buffers.empty()) {
		return;
	}

	// Publish all queued downloads before releasing their tracked pages and owners.
	const auto completion_tick = m_scheduler.CurrentTick();
	m_scheduler.Wait(completion_tick);
	m_scheduler.WaitPriorityOperations(completion_tick);
	for (const auto id: dirty_buffers) {
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.UnmarkRegionAsGpuModified(buffer.CpuAddress(), buffer.Size());
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size()) ||
		    m_gpu_modified_ranges.Intersects(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: garbage collection retained GPU ownership\n");
		}
		m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
		Unregister(id);
		m_slot_buffers.erase(id);
	}
}

void BufferCache::ProcessFaultBuffer() {
	m_fault_manager.ProcessFaultBuffer();
}

void BufferCache::SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size) {
	const auto end = vaddr + size;
	auto       it  = m_buffers.upper_bound(vaddr);
	if (it != m_buffers.begin()) {
		--it;
	}
	for (; it != m_buffers.end() && it->first < end; ++it) {
		auto&      buffer = m_slot_buffers[it->second];
		const auto start  = std::max(buffer.CpuAddress(), vaddr);
		const auto finish = std::min(buffer.CpuAddress() + buffer.Size(), end);
		if (start < finish) {
			(void)SynchronizeBuffer(buffer, start, finish - start, false, false);
		}
	}
}

void BufferCache::PublishBdaHints(uint64_t vaddr, uint64_t size) noexcept {
	m_memory_tracker.PublishBdaHints(vaddr, size);
}

void BufferCache::SynchronizeBdaLegacy(const RangeSet& mapped) {
	// Clear before the full walk so writes racing with the walk stay pending. Summaries go
	// first: a hint published in between re-sets its summary bit for the next pass.
	for (size_t summary = 0; summary < MemoryTracker::BDA_SUMMARY_WORDS; ++summary) {
		(void)m_memory_tracker.ConsumeBdaSummaryWord(summary);
	}
	for (size_t word = 0; word < MemoryTracker::BDA_HINT_WORDS; ++word) {
		(void)m_memory_tracker.ConsumeBdaHintWord(word);
	}
	mapped.ForEach(
	    [this](uint64_t begin, uint64_t end) { SynchronizeBuffersInRange(begin, end - begin); });
}

bool BufferCache::SynchronizeBdaSelective(const RangeSet& mapped) {
	for (size_t summary = 0; summary < MemoryTracker::BDA_SUMMARY_WORDS; ++summary) {
		uint64_t words = m_memory_tracker.ConsumeBdaSummaryWord(summary);
		// On failure, words not yet visited keep their hint bits; restore their summary bits.
		struct SummaryClaim {
			MemoryTracker& tracker;
			size_t         summary;
			uint64_t&      remaining;
			~SummaryClaim() { tracker.RestoreBdaSummary(summary, remaining); }
		} summary_claim {m_memory_tracker, summary, words};
		while (words != 0) {
			const auto word = summary * 64 + static_cast<size_t>(std::countr_zero(words));
			if (!SynchronizeBdaWord(word, mapped)) {
				return false;
			}
			words &= words - 1;
		}
	}
	return true;
}

bool BufferCache::SynchronizeBdaWord(size_t word, const RangeSet& mapped) {
	uint64_t bits = m_memory_tracker.ConsumeBdaHintWord(word);
	struct Claim {
		MemoryTracker& tracker;
		size_t         word;
		uint64_t&      remaining;
		~Claim() { tracker.RestoreBdaHints(word, remaining); }
	} claim {m_memory_tracker, word, bits};
	while (bits != 0) {
		const auto region = word * 64 + static_cast<size_t>(std::countr_zero(bits));
		if (!SynchronizeBdaRegion(region, mapped)) {
			return false;
		}
		// Do not touch the shared word again: a concurrent write may have republished it.
		bits &= bits - 1;
	}
	return true;
}

bool BufferCache::SynchronizeBdaRegion(uint64_t region, const RangeSet& mapped) {
	const auto region_begin = region * TRACKER_REGION_SIZE;
	auto*      manager      = m_memory_tracker.FindRegion(region);
	if (manager == nullptr) {
		// A mapping or registration can precede tracker creation. The current consumer must
		// see those bytes now; the full walk creates (or waits for) the initially dirty manager.
		mapped.ForEachInRange(region_begin, TRACKER_REGION_SIZE,
		                      [this](uint64_t begin, uint64_t end) {
			                      SynchronizeBuffersInRange(begin, end - begin);
		                      });
		return true;
	}
	const auto dirty = m_memory_tracker.SnapshotCpuDirty(*manager);
	if (dirty.None()) {
		return true;
	}
	bool consistent = true;
	mapped.ForEachInRange(region_begin, TRACKER_REGION_SIZE, [&](uint64_t begin, uint64_t end) {
		if (consistent) {
			consistent = SynchronizeDirtyOwners(dirty, region_begin, begin, end);
		}
	});
	return consistent;
}

bool BufferCache::SynchronizeDirtyOwners(const RegionBits& dirty, uint64_t region_begin,
                                         uint64_t begin, uint64_t end) {
	uint64_t   synced_end = begin;
	auto       page       = static_cast<size_t>((begin - region_begin) / TRACKER_PAGE_SIZE);
	const auto limit =
	    static_cast<size_t>((end - region_begin + TRACKER_PAGE_SIZE - 1) / TRACKER_PAGE_SIZE);
	while (page < limit) {
		const auto [first, last] = dirty.FirstRangeFrom(page);
		if (first >= limit) {
			break;
		}
		const auto run_end = std::min(region_begin + last * TRACKER_PAGE_SIZE, end);
		for (auto cursor = std::max(region_begin + first * TRACKER_PAGE_SIZE, synced_end);
		     cursor < run_end;) {
			const auto* owner = m_page_table.Find(cursor >> CACHING_PAGEBITS);
			if (owner == nullptr || !*owner) {
				cursor = Common::AlignDown(cursor, CACHING_PAGESIZE) + CACHING_PAGESIZE;
				continue;
			}
			auto* buffer = m_slot_buffers.try_get(*owner);
			if (buffer == nullptr || buffer->is_deleted || !buffer->IsInBounds(cursor, 1)) {
				return false;
			}
			const auto owner_begin = std::max(buffer->CpuAddress(), begin);
			const auto owner_end   = std::min(buffer->CpuAddress() + buffer->Size(), end);
			(void)SynchronizeBuffer(*buffer, owner_begin, owner_end - owner_begin, false, false);
			synced_end = owner_end;
			cursor     = owner_end;
		}
		page = last;
	}
	return true;
}

bool BufferCache::CheckBdaHintInvariant(const RangeSet& mapped) {
	bool covered = true;
	for (const auto& [address, id]: m_buffers) {
		const auto& buffer = m_slot_buffers[id];
		mapped.ForEachInRange(address, buffer.Size(), [&](uint64_t begin, uint64_t end) {
			covered = covered && m_memory_tracker.BdaHintsCoverCpuDirty(begin, end - begin);
		});
		if (!covered) {
			break;
		}
	}
	return covered;
}

} // namespace Libs::Graphics
