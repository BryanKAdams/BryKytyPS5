#include "graphics/host_gpu/renderer/pipeline/pipelineLibrary.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"

#include <algorithm>
#include <chrono>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace Libs::Graphics {

namespace {

// Prefetched parts compile while Thread_Gpu is stalled on a loading burst's first pipeline; six
// threads leave the game two cores of an 8-core CPU during loads, when it needs little.
constexpr uint32_t CompileThreadCount = 6;

// Background compiles run below the game's priority: while the game runs (with asynchronous
// pipelines, or during play), they use spare cores instead of slowing its frames.
void LowerThreadPriority() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#endif
}

} // namespace

PipelineLibraryCache::PipelineLibraryCache(GraphicContext& graphics, vk::PipelineCache driver_cache)
    : m_graphics(graphics), m_driver_cache(driver_cache),
      m_link_thread([this](const std::stop_token& stop) { LinkThread(stop); }) {
	for (uint32_t i = 0; i < CompileThreadCount; i++) {
		m_compile_threads.emplace_back([this](const std::stop_token& stop) { CompileThread(stop); });
	}
}

PipelineLibraryCache::~PipelineLibraryCache() {
	Stop();
	const auto& device = m_graphics.device;
	for (const auto& [target, pipeline]: m_optimized) {
		(void)target;
		if (pipeline != nullptr) {
			device.destroyPipeline(pipeline, nullptr);
		}
	}
	for (auto& [key, entry]: m_libraries) {
		(void)key;
		// Stop resolved every queued job, and the threads are joined.
		const auto pipeline = entry.pending.valid() ? entry.pending.get() : entry.pipeline;
		if (pipeline != nullptr) {
			device.destroyPipeline(pipeline, nullptr);
		}
	}
	for (const auto layout: m_kept_layouts) {
		device.destroyPipelineLayout(layout, nullptr);
	}
	for (const auto set_layout: m_kept_set_layouts) {
		device.destroyDescriptorSetLayout(set_layout, nullptr);
	}
}

PipelineLibraryCache::Found PipelineLibraryCache::Find(const std::string& key) {
	const auto iter = m_libraries.find(key);
	if (iter == m_libraries.end()) {
		return {};
	}
	auto& entry = iter->second;
	if (entry.pending.valid()) {
		KYTY_PROFILER_BLOCK("PipelineLibraryCache::WaitForPrefetch");
		entry.pipeline = entry.pending.get();
		entry.pending  = {};
		if (entry.pipeline == nullptr) {
			// The prefetch failed or was dropped; the caller compiles the part itself.
			m_libraries.erase(iter);
			return {};
		}
	}
	return {entry.pipeline, entry.prefetched};
}

bool PipelineLibraryCache::Ready(const std::string& key) const {
	const auto iter = m_libraries.find(key);
	if (iter == m_libraries.end()) {
		return false;
	}
	const auto& pending = iter->second.pending;
	return !pending.valid() || pending.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

vk::Pipeline PipelineLibraryCache::Take(const std::string& key) {
	const auto found = Find(key);
	m_libraries.erase(key);
	return found.pipeline;
}

vk::Pipeline PipelineLibraryCache::Insert(std::string key, vk::Pipeline library) {
	EXIT_IF(library == nullptr);
	const auto [iter, inserted] = m_libraries.emplace(std::move(key), Entry {library, {}, false});
	EXIT_IF(!inserted);
	return iter->second.pipeline;
}

bool PipelineLibraryCache::Prefetch(std::string key, Common::UniqueFunction<vk::Pipeline>&& compile,
                                    bool urgent) {
	if (m_libraries.contains(key)) {
		return false;
	}
	CompileJob job;
	job.key     = key;
	job.compile = std::move(compile);
	Entry entry;
	entry.pending    = job.result.get_future().share();
	entry.prefetched = true;
	{
		std::lock_guard lock(m_compile_mutex);
		if (m_compile_stopped) {
			return false;
		}
		if (urgent) {
			m_compile_jobs.push_front(std::move(job));
		} else {
			m_compile_jobs.push_back(std::move(job));
		}
	}
	m_libraries.emplace(std::move(key), std::move(entry));
	m_compile_available.notify_one();
	return true;
}

void PipelineLibraryCache::Promote(const std::string& key) {
	std::lock_guard lock(m_compile_mutex);
	const auto      queued = std::ranges::find(m_compile_jobs, key, &CompileJob::key);
	if (queued != m_compile_jobs.end() && queued != m_compile_jobs.begin()) {
		auto job = std::move(*queued);
		m_compile_jobs.erase(queued);
		m_compile_jobs.push_front(std::move(job));
	}
}

void PipelineLibraryCache::KeepLayout(vk::PipelineLayout                       layout,
                                      std::span<const vk::DescriptorSetLayout> set_layouts) {
	m_kept_layouts.push_back(layout);
	m_kept_set_layouts.insert(m_kept_set_layouts.end(), set_layouts.begin(), set_layouts.end());
}

void PipelineLibraryCache::QueueOptimizedLink(const void*                   target,
                                              std::span<const vk::Pipeline> parts,
                                              vk::PipelineLayout            layout) {
	LinkJob job;
	EXIT_IF(parts.size() > job.parts.size());
	job.target     = target;
	job.part_count = static_cast<uint32_t>(parts.size());
	job.layout     = layout;
	std::ranges::copy(parts, job.parts.begin());
	{
		std::lock_guard lock(m_link_mutex);
		if (m_link_stopped) {
			// Shutting down: the fast-linked pipeline stays.
			m_optimized.emplace(target, nullptr);
			return;
		}
		m_link_jobs.push_back(job);
	}
	m_link_available.notify_one();
}

std::optional<vk::Pipeline> PipelineLibraryCache::TakeOptimized(const void* target) {
	std::lock_guard lock(m_link_mutex);
	const auto      iter = m_optimized.find(target);
	if (iter == m_optimized.end()) {
		return std::nullopt;
	}
	const auto pipeline = iter->second;
	m_optimized.erase(iter);
	return pipeline;
}

void PipelineLibraryCache::Stop() {
	{
		std::lock_guard lock(m_link_mutex);
		m_link_stopped = true;
		for (const auto& job: m_link_jobs) {
			m_optimized.emplace(job.target, nullptr);
		}
		m_link_jobs.clear();
	}
	{
		std::lock_guard lock(m_compile_mutex);
		m_compile_stopped = true;
		for (auto& job: m_compile_jobs) {
			job.result.set_value(nullptr);
		}
		m_compile_jobs.clear();
	}
	m_link_thread.request_stop();
	for (auto& thread: m_compile_threads) {
		thread.request_stop();
	}
	if (m_link_thread.joinable()) {
		m_link_thread.join();
	}
	for (auto& thread: m_compile_threads) {
		if (thread.joinable()) {
			thread.join();
		}
	}
}

void PipelineLibraryCache::LinkThread(const std::stop_token& stop) {
	KYTY_PROFILER_THREAD("PipelineLink");
	LowerThreadPriority();
	for (;;) {
		LinkJob job;
		{
			std::unique_lock lock(m_link_mutex);
			if (!m_link_available.wait(lock, stop, [this] { return !m_link_jobs.empty(); })) {
				return;
			}
			job = m_link_jobs.front();
			m_link_jobs.pop_front();
		}
		vk::PipelineLibraryCreateInfoKHR libraries {};
		libraries.libraryCount = job.part_count;
		libraries.pLibraries   = job.parts.data();
		vk::GraphicsPipelineCreateInfo info {};
		info.pNext  = &libraries;
		info.flags  = vk::PipelineCreateFlagBits::eLinkTimeOptimizationEXT;
		info.layout = job.layout;
		vk::Pipeline optimized = nullptr;
		const auto   result =
		    m_graphics.device.createGraphicsPipelines(m_driver_cache, 1, &info, nullptr, &optimized);
		if (result != vk::Result::eSuccess) {
			LOGF("PipelineLibrary: optimized link failed (%s); keeping the fast-linked pipeline\n",
			     vk::to_string(result).c_str());
			optimized = nullptr;
		}
		std::lock_guard lock(m_link_mutex);
		m_optimized.emplace(job.target, optimized);
	}
}

void PipelineLibraryCache::CompileThread(const std::stop_token& stop) {
	KYTY_PROFILER_THREAD("PipelinePrefetch");
	LowerThreadPriority();
	for (;;) {
		CompileJob job;
		{
			std::unique_lock lock(m_compile_mutex);
			if (!m_compile_available.wait(lock, stop, [this] { return !m_compile_jobs.empty(); })) {
				return;
			}
			job = std::move(m_compile_jobs.front());
			m_compile_jobs.pop_front();
		}
		job.result.set_value(job.compile());
	}
}

} // namespace Libs::Graphics
