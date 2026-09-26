#include "graphics/host_gpu/renderer/pipeline/pipelineLibrary.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"

#include <algorithm>

namespace Libs::Graphics {

PipelineLibraryCache::PipelineLibraryCache(GraphicContext& graphics, vk::PipelineCache driver_cache)
    : m_graphics(graphics), m_driver_cache(driver_cache),
      m_link_thread([this](const std::stop_token& stop) { LinkThread(stop); }) {}

PipelineLibraryCache::~PipelineLibraryCache() {
	Stop();
	const auto& device = m_graphics.device;
	for (const auto& [target, pipeline]: m_optimized) {
		(void)target;
		if (pipeline != nullptr) {
			device.destroyPipeline(pipeline, nullptr);
		}
	}
	for (const auto& [key, library]: m_libraries) {
		(void)key;
		device.destroyPipeline(library, nullptr);
	}
}

vk::Pipeline PipelineLibraryCache::Find(const std::string& key) const {
	const auto iter = m_libraries.find(key);
	return iter != m_libraries.end() ? iter->second : nullptr;
}

vk::Pipeline PipelineLibraryCache::Insert(std::string key, vk::Pipeline library) {
	EXIT_IF(library == nullptr);
	const auto [iter, inserted] = m_libraries.emplace(std::move(key), library);
	EXIT_IF(!inserted);
	return iter->second;
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
		if (m_stopped) {
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
		m_stopped = true;
		for (const auto& job: m_link_jobs) {
			m_optimized.emplace(job.target, nullptr);
		}
		m_link_jobs.clear();
	}
	m_link_thread.request_stop();
	if (m_link_thread.joinable()) {
		m_link_thread.join();
	}
}

void PipelineLibraryCache::LinkThread(const std::stop_token& stop) {
	KYTY_PROFILER_THREAD("PipelineLink");
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

} // namespace Libs::Graphics
