#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINELIBRARY_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINELIBRARY_H_

#include "common/common.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;

// VK_EXT_graphics_pipeline_library parts (vertex input, pre-rasterization shaders, fragment
// shader, fragment output), each cached under the state it depends on. A new graphics pipeline
// compiles only the parts no earlier pipeline built, then links them in well under a millisecond.
// A background thread relinks the pipeline with link-time optimization, and the draw path swaps
// that pipeline in once it is ready.
//
// Parts can also be prefetched: a look-ahead over the rest of a command buffer predicts the parts
// of upcoming draws and compiles them on worker threads, and Find waits for a part still compiling.
class PipelineLibraryCache {
public:
	PipelineLibraryCache(GraphicContext& graphics, vk::PipelineCache driver_cache);
	~PipelineLibraryCache();
	KYTY_CLASS_NO_COPY(PipelineLibraryCache);

	struct Found {
		vk::Pipeline pipeline   = nullptr;
		// Compiled by a prefetch, not by the draw that uses it first.
		bool         prefetched = false;
	};

	// `key` starts with the part and holds everything that part's create info depends on. Find
	// waits for a part a prefetch is still compiling, and returns null for a part not built (or a
	// prefetch that failed). Called under the pipeline cache's lock, like the rest of this class.
	[[nodiscard]] Found Find(const std::string& key);
	vk::Pipeline        Insert(std::string key, vk::Pipeline library);
	[[nodiscard]] bool  Contains(const std::string& key) const { return m_libraries.contains(key); }

	// Compiles a part on a worker thread under `key`, unless the key is already present. `compile`
	// owns everything its create info points to. Returns whether a job was queued.
	bool Prefetch(std::string key, Common::UniqueFunction<vk::Pipeline>&& compile);
	// Layouts a prefetch created its parts with; destroyed with the cache.
	void KeepLayout(vk::PipelineLayout layout, std::span<const vk::DescriptorSetLayout> set_layouts);

	// Queues a link-time-optimized link of `parts` with `layout`; `target` names the result.
	// `layout` must stay alive until the result is taken or the thread stops.
	void QueueOptimizedLink(const void* target, std::span<const vk::Pipeline> parts,
	                        vk::PipelineLayout layout);
	// Nothing while the link is queued or running; then the optimized pipeline, which the caller
	// now owns, or null when the link failed and the fast-linked pipeline stays.
	[[nodiscard]] std::optional<vk::Pipeline> TakeOptimized(const void* target);
	// Stops the link and compile threads and drops queued work. Must run before the driver cache
	// is destroyed.
	void Stop();

private:
	struct Entry {
		vk::Pipeline                     pipeline = nullptr;
		std::shared_future<vk::Pipeline> pending;
		bool                             prefetched = false;
	};

	struct LinkJob {
		const void*                 target = nullptr;
		// Three parts for mesh pipelines, which have no vertex input part.
		std::array<vk::Pipeline, 4> parts {};
		uint32_t                    part_count = 0;
		vk::PipelineLayout          layout     = nullptr;
	};

	struct CompileJob {
		std::promise<vk::Pipeline>           result;
		Common::UniqueFunction<vk::Pipeline> compile;
	};

	void LinkThread(const std::stop_token& stop);
	void CompileThread(const std::stop_token& stop);

	GraphicContext&                        m_graphics;
	vk::PipelineCache                      m_driver_cache = nullptr;
	std::unordered_map<std::string, Entry> m_libraries;
	std::vector<vk::PipelineLayout>        m_kept_layouts;
	std::vector<vk::DescriptorSetLayout>   m_kept_set_layouts;

	std::mutex                                    m_link_mutex;
	std::condition_variable_any                   m_link_available;
	std::deque<LinkJob>                           m_link_jobs;
	std::unordered_map<const void*, vk::Pipeline> m_optimized;
	bool                                          m_link_stopped = false;

	std::mutex                  m_compile_mutex;
	std::condition_variable_any m_compile_available;
	std::deque<CompileJob>      m_compile_jobs;
	bool                        m_compile_stopped = false;

	std::jthread              m_link_thread;
	std::vector<std::jthread> m_compile_threads;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINELIBRARY_H_
