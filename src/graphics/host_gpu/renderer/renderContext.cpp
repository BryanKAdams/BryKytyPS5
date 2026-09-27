#include "graphics/host_gpu/renderer/renderContext.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/drainStats.h"
#include "graphics/presentation/videoOut.h"
#include "libs/errno.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics {

RenderContext::RenderContext(GraphicContext& graphics)
    : m_graphics(graphics), m_render_executor(*this), m_command_scheduler(*this, graphics),
      m_descriptor_heap(graphics, m_command_scheduler.GetMasterSemaphore()),
      m_pipeline_cache(graphics), m_sampler_cache(graphics),
      m_buffer_cache(graphics, m_command_scheduler, m_page_manager, m_texture_cache),
      m_texture_cache(graphics, m_command_scheduler, m_page_manager, m_buffer_cache) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	if (Config::AsyncSubmitEnabled()) {
		// vkQueueSubmit was about 40% of Thread_Gpu in Astro Bot; a queue thread takes it over.
		m_command_scheduler.EnableAsyncSubmit();
	}
}

RenderContext::~RenderContext() {
	ShutdownGpu();
	m_command_scheduler.Shutdown();
}

void RenderContext::InitializeGpu(VideoOut::VideoOutDriver* video_out) {
	EXIT_IF(m_gpu != nullptr);
	m_video_out = video_out;
	m_gpu       = std::make_unique<GuestGpu>(*this);
}

void RenderContext::ShutdownGpu() {
	if (m_gpu != nullptr) {
		m_gpu->Shutdown();
		m_gpu.reset();
	}
	if (m_video_out != nullptr) {
		if (m_command_scheduler.Active()) {
			m_command_scheduler.Finish();
		}
		m_command_scheduler.DrainPriorityOperations();
		m_video_out = nullptr;
	}
}

GuestGpu& RenderContext::GetGpu() const {
	EXIT_IF(m_gpu == nullptr);
	return *m_gpu;
}

VideoOut::VideoOutDriver& RenderContext::GetVideoOut() const {
	EXIT_IF(m_video_out == nullptr);
	return *m_video_out;
}

bool RenderContext::HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept {
	// The host reports the faulting byte, not the instruction's access width. Both caches
	// resolve its page; guessing a width can cross the end of a valid guest mapping.
	constexpr uint64_t fault_size = 1;
	if (!IsMapped(fault_vaddr, fault_size)) {
		return false;
	}
	const bool gpu_thread = GuestGpu::IsGpuThread();
	struct SiteTimer {
		bool                                  enabled;
		bool                                  write;
		uint64_t                              address;
		std::chrono::steady_clock::time_point start;
		~SiteTimer() {
			if (enabled) {
				const auto elapsed = std::chrono::steady_clock::now() - start;
				DrainStats::RecordFaultSite(
				    DrainStats::t_fault_pc, address, write,
				    static_cast<uint64_t>(
				        std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()));
			}
		}
	} site_timer {DrainStats::Enabled(), access == PageFaultAccess::Write, fault_vaddr,
	              DrainStats::Enabled() ? std::chrono::steady_clock::now()
	                                    : std::chrono::steady_clock::time_point {}};
	if (access == PageFaultAccess::Write) {
		DrainStats::ReasonScope reason(gpu_thread ? DrainStats::Reason::GpuThreadWriteFault
		                                          : DrainStats::Reason::GuestWriteFault);
		RecordUpload(UploadSource::Fault, fault_vaddr, 0x1000);
		m_buffer_cache.InvalidateMemory(fault_vaddr, fault_size);
		m_texture_cache.InvalidateMemory(fault_vaddr, fault_size);
		if (gpu_thread) {
			// The command stream writes this (WRITE_DATA, fills): later draws must see it.
			AdvanceBdaEpoch();
		}
	} else {
		DrainStats::ReasonScope reason(gpu_thread ? DrainStats::Reason::GpuThreadReadFault
		                                          : DrainStats::Reason::GuestReadFault);
		m_buffer_cache.ReadMemory(fault_vaddr, fault_size);
	}
	return true;
}

bool RenderContext::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (!IsMapped(vaddr, size)) {
		return false;
	}
	DrainStats::ReasonScope reason(DrainStats::Reason::KernelInvalidate);
	m_buffer_cache.InvalidateMemory(vaddr, size);
	m_texture_cache.InvalidateMemory(vaddr, size);
	AdvanceBdaEpoch();
	return true;
}

bool RenderContext::IsMapped(uint64_t vaddr, uint64_t size) const noexcept {
	if (!GuestRange {vaddr, size}.Valid()) {
		return false;
	}
	std::shared_lock lock(m_mapped_ranges_mutex);
	return m_mapped_ranges.Contains(vaddr, size);
}

void RenderContext::MapMemory(uint64_t vaddr, uint64_t size) {
	std::lock_guard lock(m_mapped_ranges_mutex);
	m_mapped_ranges.Add(vaddr, size);
	m_buffer_cache.PublishBdaHints(vaddr, size);
	AdvanceBdaEpoch();
}

void RenderContext::UnmapMemory(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		return;
	}
	{
		// Memory the GPU never mapped holds no cache entries: HandleFault and
		// InvalidateMemory gate on the same set. Skip the full GPU drain for it.
		std::shared_lock lock(m_mapped_ranges_mutex);
		if (!m_mapped_ranges.Intersects(vaddr, size)) {
			return;
		}
	}
	if (CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported memory unmap from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	const auto unmap = [this, vaddr, size] {
		DrainStats::ReasonScope reason(DrainStats::Reason::Unmap);
		if (m_command_scheduler.Active()) {
			const auto tick = m_command_scheduler.CurrentTick();
			m_command_scheduler.Finish();
			m_command_scheduler.WaitPriorityOperations(tick);
		}
		m_buffer_cache.InvalidateMemory(vaddr, size);
		m_texture_cache.UnmapMemory(vaddr, size);
		std::lock_guard lock(m_mapped_ranges_mutex);
		m_mapped_ranges.Subtract(vaddr, size);
		AdvanceBdaEpoch();
	};
	// Shutdown still owns the GPU while queued rendering drains, but its command lane no
	// longer accepts external work. Use the guest GPU's state for the teardown route.
	if (m_gpu == nullptr || m_gpu->IsStopping()) {
		unmap();
		return;
	}
	m_gpu->SendCommandSync(unmap);
}

// KYTY_DEBUG_BDA_EPOCH=0 synchronizes before every draw that reads memory through addresses;
// KYTY_DEBUG_AB=bdaepoch alternates.
static bool BdaEpochEnabled() {
	static const bool enabled = [] {
		const char* text = std::getenv("KYTY_DEBUG_BDA_EPOCH");
		return text == nullptr || std::strcmp(text, "0") != 0;
	}();
	static const bool ab = AbSelected("bdaepoch");
	return enabled && !(ab && AbFeatureOff());
}

// Draws reading memory through addresses must see the CPU writes made before their submission,
// so the first such draw of each epoch uploads every CPU-dirty page. Meanwhile guest threads keep
// writing later frames' data (Astro Bot's Sky Garden faults about 150k pages/s): uploading and
// re-protecting those pages before every such draw only brings the next fault sooner, since no
// draw of this submission may read them. Everything that can make newer CPU writes visible to
// later draws starts an epoch: a submission starting or resuming, packets that read guest memory
// (waits, conditions, predication), new page-table entries (buffer registration), GPU mappings,
// kernel invalidations and the GPU thread's own writes.
void RenderContext::PrepareBda() {
	RecordUpload(UploadSource::BdaPass, 0, 0);
	m_fault_process_pending = true;
	const auto epoch        = m_bda_epoch.load(std::memory_order_acquire);
	if (epoch == m_bda_synced_epoch && BdaEpochEnabled()) {
		return;
	}
	// An epoch started during the synchronization below leaves the next call synchronizing.
	m_bda_synced_epoch = epoch;
	RecordUpload(UploadSource::BdaSync, 0, 0);
	std::shared_lock lock(m_mapped_ranges_mutex);
	const auto       mode = Config::GetBdaSyncMode();
	if (mode == Config::BdaSyncMode::Legacy ||
	    !m_buffer_cache.SynchronizeBdaSelective(m_mapped_ranges)) {
		m_buffer_cache.SynchronizeBdaLegacy(m_mapped_ranges);
	}
	if (mode == Config::BdaSyncMode::SelectiveChecked) {
		EXIT_IF(!m_buffer_cache.CheckBdaHintInvariant(m_mapped_ranges));
	}
}

void RenderContext::RunGarbageCollector() {
	if (m_fault_process_pending) {
		m_fault_process_pending = false;
		m_buffer_cache.ProcessFaultBuffer();
	}
	m_texture_cache.ProcessDownloadImages();
	m_texture_cache.RunGarbageCollector();
	m_buffer_cache.RunGarbageCollector();
}

void RenderContext::AddInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it != m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.push_back({eq, event_id});
}

void RenderContext::DeleteInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it == m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.erase(it);
}

void RenderContext::TriggerInterrupt(int event_id, uint32_t context_id) {
	std::vector<InterruptEqRegistration> registrations;
	{
		Common::LockGuard lock(m_interrupt_mutex);
		for (const auto& registration: m_interrupt_eqs) {
			if (registration.event_id == event_id) {
				registrations.push_back(registration);
			}
		}
	}

	for (const auto& registration: registrations) {
		const auto result = LibKernel::EventQueue::KernelTriggerEvent(
		    registration.eq, static_cast<uintptr_t>(registration.event_id),
		    LibKernel::EventQueue::KERNEL_EVFILT_GRAPHICS,
		    reinterpret_cast<void*>(static_cast<uintptr_t>(context_id)));
		if (result == LibKernel::KERNEL_ERROR_EBADF || result == LibKernel::KERNEL_ERROR_ENOENT) {
			DeleteInterruptEq(registration.eq, registration.event_id);
			continue;
		}
		EXIT_NOT_IMPLEMENTED(result != OK);
	}
}

} // namespace Libs::Graphics
