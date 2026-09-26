#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINELIBRARY_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINELIBRARY_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>

namespace Libs::Graphics {

struct GraphicContext;

// VK_EXT_graphics_pipeline_library parts (vertex input, pre-rasterization shaders, fragment
// shader, fragment output), each cached under the state it depends on. A new graphics pipeline
// compiles only the parts no earlier pipeline built, then links them in well under a millisecond.
// A background thread relinks the pipeline with link-time optimization, and the draw path swaps
// that pipeline in once it is ready.
class PipelineLibraryCache {
public:
	PipelineLibraryCache(GraphicContext& graphics, vk::PipelineCache driver_cache);
	~PipelineLibraryCache();
	KYTY_CLASS_NO_COPY(PipelineLibraryCache);

	// `key` starts with the part and holds everything that part's create info depends on; Find
	// returns null for a part not built yet. Called under the pipeline cache's lock.
	[[nodiscard]] vk::Pipeline Find(const std::string& key) const;
	vk::Pipeline               Insert(std::string key, vk::Pipeline library);

	// Queues a link-time-optimized link of `parts` with `layout`; `target` names the result.
	// `layout` must stay alive until the result is taken or the thread stops.
	void QueueOptimizedLink(const void* target, std::span<const vk::Pipeline> parts,
	                        vk::PipelineLayout layout);
	// Nothing while the link is queued or running; then the optimized pipeline, which the caller
	// now owns, or null when the link failed and the fast-linked pipeline stays.
	[[nodiscard]] std::optional<vk::Pipeline> TakeOptimized(const void* target);
	// Stops the link thread and drops queued links. Must run before the driver cache is destroyed.
	void Stop();

private:
	struct LinkJob {
		const void*                 target = nullptr;
		// Three parts for mesh pipelines, which have no vertex input part.
		std::array<vk::Pipeline, 4> parts {};
		uint32_t                    part_count = 0;
		vk::PipelineLayout          layout     = nullptr;
	};

	void LinkThread(const std::stop_token& stop);

	GraphicContext&                          m_graphics;
	vk::PipelineCache                        m_driver_cache = nullptr;
	std::unordered_map<std::string, vk::Pipeline> m_libraries;

	std::mutex                                        m_link_mutex;
	std::condition_variable_any                       m_link_available;
	std::deque<LinkJob>                               m_link_jobs;
	std::unordered_map<const void*, vk::Pipeline>     m_optimized;
	bool                                              m_stopped = false;
	std::jthread                                      m_link_thread;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINELIBRARY_H_
