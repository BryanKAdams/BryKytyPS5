#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERCONTEXT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERCONTEXT_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/threads.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/samplerCache.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/pipeline/descriptorHeap.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "kernel/eventQueue.h"

#include <array>
#include <atomic>
#include <memory>
#include <shared_mutex>
#include <vector>

namespace Libs::VideoOut {
class VideoOutDriver;
}

namespace Libs::Graphics {

class GuestGpu;

// The game re-read the same large file several times in a row: a texture streamer asked to keep
// more than it can fit (see RenderContext::ReportMipStats).
void NoteStreamingThrash() noexcept;

class RenderContext {
public:
	explicit RenderContext(GraphicContext& graphics);
	~RenderContext();
	KYTY_CLASS_NO_COPY(RenderContext);

	[[nodiscard]] GraphicContext&           GetGraphics() const noexcept { return m_graphics; }
	void                                    InitializeGpu(VideoOut::VideoOutDriver* video_out);
	void                                    ShutdownGpu();
	[[nodiscard]] GuestGpu&                 GetGpu() const;
	[[nodiscard]] VideoOut::VideoOutDriver& GetVideoOut() const;

	Common::Mutex&      GetMutex() { return m_mutex; }
	CommandScheduler&   GetCommandScheduler() { return m_command_scheduler; }
	PipelineCache&      GetPipelineCache() { return m_pipeline_cache; }
	DescriptorHeap&     GetDescriptorHeap() { return m_descriptor_heap; }
	SamplerCache&       GetSamplerCache() { return m_sampler_cache; }
	BufferCache&        GetBufferCache() { return m_buffer_cache; }
	TextureCache&       GetTextureCache() { return m_texture_cache; }
	RenderExecutor&     GetRenderExecutor() { return m_render_executor; }

	[[nodiscard]] bool HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept;
	[[nodiscard]] bool InvalidateMemory(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsMapped(uint64_t vaddr, uint64_t size) const noexcept;
	void               MapMemory(uint64_t vaddr, uint64_t size);
	void               UnmapMemory(uint64_t vaddr, uint64_t size);
	void               PrepareBda();
	// Starts a new epoch for PrepareBda (see there): the next draw that reads memory through
	// addresses uploads every CPU write made so far.
	void AdvanceBdaEpoch() noexcept { m_bda_epoch.fetch_add(1, std::memory_order_release); }
	// The GPU thread wrote guest memory itself: a page that is not write-protected takes the
	// write without a fault, so the write starts an epoch.
	void AdvanceBdaEpochForGpuWrite() noexcept;
	// Counts a draw's sampled texture in its mip statistics counter (the descriptor's
	// MIP_STATS_CNT_ID) for the next GET_LOD_STATS report (Thread_Gpu only).
	void MarkMipStatsCounter(uint32_t id) noexcept {
		m_mip_stats_counters[(id / 64u) & 3u] |= uint64_t {1} << (id % 64u);
	}
	// Writes a GET_LOD_STATS report of `size` bytes to `dst` (see there), then starts counting
	// anew if `reset`.
	void ReportMipStats(void* dst, uint32_t size, bool reset);
	void RunGarbageCollector();

	void AddInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id);
	void DeleteInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id);
	void TriggerInterrupt(int event_id, uint32_t context_id);

private:
	struct InterruptEqRegistration {
		LibKernel::EventQueue::KernelEqueue eq       = LibKernel::EventQueue::KERNEL_EQUEUE_INVALID;
		int                                 event_id = 0;
	};

	GraphicContext&           m_graphics;
	Common::Mutex             m_mutex;
	RenderExecutor            m_render_executor;
	CommandScheduler          m_command_scheduler;
	DescriptorHeap            m_descriptor_heap;
	PipelineCache             m_pipeline_cache;
	SamplerCache              m_sampler_cache;
	PageManager               m_page_manager;
	BufferCache               m_buffer_cache;
	TextureCache              m_texture_cache;
	mutable std::shared_mutex m_mapped_ranges_mutex;
	RangeSet                  m_mapped_ranges;
	std::unique_ptr<GuestGpu> m_gpu;
	VideoOut::VideoOutDriver* m_video_out = nullptr;
	bool                      m_fault_process_pending = false;
	// PrepareBda's current epoch and the epoch of its last synchronization (Thread_Gpu only).
	std::atomic<uint64_t>     m_bda_epoch {1};
	uint64_t                  m_bda_synced_epoch = 0;
	// Mip statistics (Thread_Gpu only): the counters sampled since the last report, one bit each,
	// the reports made, the last report in which each counter, and any, was sampled, and when the
	// current relief from a streaming thrash ends (0: none) and the next may start (steady_clock
	// ticks).
	std::array<uint64_t, 4>   m_mip_stats_counters {};
	uint64_t                  m_mip_stats_reports         = 0;
	uint64_t                  m_mip_stats_last_marked_any = 0;
	std::array<uint64_t, 256> m_mip_stats_last_marked {};
	int64_t                   m_mip_stats_relief_end = 0;
	int64_t                   m_mip_stats_rearm_time = 0;

	Common::Mutex                        m_interrupt_mutex;
	std::vector<InterruptEqRegistration> m_interrupt_eqs;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERCONTEXT_H_
