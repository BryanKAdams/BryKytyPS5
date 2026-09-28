#include "graphics/host_gpu/vulkanCommon.h"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-completeness"
#pragma clang diagnostic ignored "-Wunused-private-field"
#pragma clang diagnostic ignored "-Wunused-variable"
#endif

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/debug.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <mutex>

namespace Libs::Graphics {

namespace {

// VMA fetches the memory budget from the driver again once 30 allocations or frees have passed
// since the last fetch (VmaAllocator_T::GetHeapBudgets), and a fetch took the AMD driver about a
// millisecond: with Sky Garden's buffer and image churn, about 1% of the GPU thread. Budget
// queries are answered from a fetch younger than FetchInterval instead, with each heap's usage
// moved by the device memory VMA allocated and freed since, as VMA itself estimates usage between
// its fetches. Only the other processes' usage is then up to FetchInterval old.
// KYTY_DEBUG_AB=budgetcache fetches every time in every other window.
struct BudgetCache {
	static constexpr auto FetchInterval = std::chrono::milliseconds(500);

	std::mutex                                            mutex;
	PFN_vkGetPhysicalDeviceMemoryProperties2              fetch = nullptr;
	bool                                                  valid = false;
	std::chrono::steady_clock::time_point                 fetched;
	VkPhysicalDeviceMemoryProperties                      memory {};
	std::array<VkDeviceSize, VK_MAX_MEMORY_HEAPS>         usage {};
	std::array<VkDeviceSize, VK_MAX_MEMORY_HEAPS>         budget {};
	std::array<int64_t, VK_MAX_MEMORY_HEAPS>              allocated_at_fetch {};
	// VMA's device memory per heap, kept by its allocate and free callbacks.
	std::array<std::atomic<int64_t>, VK_MAX_MEMORY_HEAPS> allocated {};
	std::array<uint32_t, VK_MAX_MEMORY_TYPES>             type_heap {};
};

BudgetCache g_budget_cache;

void VKAPI_PTR CountAllocation(VmaAllocator /*allocator*/, uint32_t memory_type,
                               VkDeviceMemory /*memory*/, VkDeviceSize size, void* /*user_data*/) {
	g_budget_cache.allocated[g_budget_cache.type_heap[memory_type]].fetch_add(
	    static_cast<int64_t>(size), std::memory_order_relaxed);
}

void VKAPI_PTR CountFree(VmaAllocator /*allocator*/, uint32_t memory_type, VkDeviceMemory /*memory*/,
                         VkDeviceSize size, void* /*user_data*/) {
	g_budget_cache.allocated[g_budget_cache.type_heap[memory_type]].fetch_sub(
	    static_cast<int64_t>(size), std::memory_order_relaxed);
}

void VKAPI_PTR CachedMemoryProperties2(VkPhysicalDevice                     physical_device,
                                       VkPhysicalDeviceMemoryProperties2* properties) {
	auto& cache  = g_budget_cache;
	auto* budget = static_cast<VkPhysicalDeviceMemoryBudgetPropertiesEXT*>(properties->pNext);
	static const bool ab = AbSelected("budgetcache");
	if (budget == nullptr ||
	    budget->sType != VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT ||
	    budget->pNext != nullptr || (ab && AbFeatureOff())) {
		cache.fetch(physical_device, properties);
		return;
	}
	std::scoped_lock lock(cache.mutex);
	const auto       now = std::chrono::steady_clock::now();
	if (!cache.valid || now - cache.fetched >= BudgetCache::FetchInterval) {
		cache.fetch(physical_device, properties);
		cache.memory = properties->memoryProperties;
		for (uint32_t heap = 0; heap < VK_MAX_MEMORY_HEAPS; heap++) {
			cache.usage[heap]              = budget->heapUsage[heap];
			cache.budget[heap]             = budget->heapBudget[heap];
			cache.allocated_at_fetch[heap] = cache.allocated[heap].load(std::memory_order_relaxed);
		}
		cache.fetched = now;
		cache.valid   = true;
		return;
	}
	properties->memoryProperties = cache.memory;
	for (uint32_t heap = 0; heap < VK_MAX_MEMORY_HEAPS; heap++) {
		const auto delta =
		    cache.allocated[heap].load(std::memory_order_relaxed) - cache.allocated_at_fetch[heap];
		const auto usage = static_cast<int64_t>(cache.usage[heap]) + delta;
		budget->heapUsage[heap]  = static_cast<VkDeviceSize>(std::max<int64_t>(usage, 0));
		budget->heapBudget[heap] = cache.budget[heap];
	}
}

} // namespace

bool GraphicContext::CreateAllocator() {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(instance == nullptr || physical_device == nullptr || device == nullptr ||
	        allocator != nullptr);

	VmaVulkanFunctions functions {};
	functions.vkGetInstanceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr;
	functions.vkGetDeviceProcAddr   = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;

	VmaAllocatorCreateInfo info {};
	info.instance         = instance;
	info.physicalDevice   = physical_device;
	info.device           = device;
	info.pVulkanFunctions = &functions;
	info.vulkanApiVersion = VULKAN_TARGET_API_VERSION;
	info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
	VmaDeviceMemoryCallbacks memory_callbacks {};
	if (memory_budget_ext_enabled) {
		info.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
		auto& cache = g_budget_cache;
		cache.fetch = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties2>(
		    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr(
		        static_cast<VkInstance>(instance), "vkGetPhysicalDeviceMemoryProperties2"));
		EXIT_IF(cache.fetch == nullptr);
		cache.valid = false;
		const auto& properties = physical_device.getMemoryProperties();
		for (uint32_t type = 0; type < properties.memoryTypeCount; type++) {
			cache.type_heap[type] = properties.memoryTypes[type].heapIndex;
		}
		functions.vkGetPhysicalDeviceMemoryProperties2KHR = CachedMemoryProperties2;
		memory_callbacks.pfnAllocate = CountAllocation;
		memory_callbacks.pfnFree     = CountFree;
		info.pDeviceMemoryCallbacks  = &memory_callbacks;
	}

	const auto result = static_cast<vk::Result>(vmaCreateAllocator(&info, &allocator));
	if (result != vk::Result::eSuccess) {
		LOGF("vmaCreateAllocator failed: %s\n", vk::to_string(result).c_str());
		return false;
	}
	return true;
}

void GraphicContext::DestroyAllocator() {
	if (allocator == nullptr) {
		return;
	}
	vmaDestroyAllocator(allocator);
	allocator = nullptr;
}

void GraphicContext::LogMemoryBudget() const {
	if (allocator == nullptr || physical_device == nullptr) {
		return;
	}

	const auto& properties = GetPhysicalDeviceMemoryProperties();
	VmaBudget   budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	for (uint32_t i = 0; i < properties.memoryHeapCount; i++) {
		LOGF("VMA heap %u: usage=%" PRIu64 ", budget=%" PRIu64 ", allocation=%" PRIu64
		     ", blocks=%" PRIu64 "\n",
		     i, static_cast<uint64_t>(budgets[i].usage), static_cast<uint64_t>(budgets[i].budget),
		     static_cast<uint64_t>(budgets[i].statistics.allocationBytes),
		     static_cast<uint64_t>(budgets[i].statistics.blockBytes));
	}
}

uint64_t GraphicContext::GetDeviceMemoryUsage() const {
	if (!CanReportMemoryUsage() || allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const bool discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	uint64_t usage = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const bool device_local =
		    static_cast<bool>(physical_device_memory_properties.memoryHeaps[heap].flags &
		                      vk::MemoryHeapFlagBits::eDeviceLocal);
		if (!discrete || device_local) {
			usage += budgets[heap].usage;
		}
	}
	return usage;
}

uint64_t GraphicContext::GetTotalMemoryBudget() const {
	if (allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const bool discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	uint64_t budget = 0;
	uint64_t local  = 0;
	uint64_t usage  = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const auto& properties = physical_device_memory_properties.memoryHeaps[heap];
		const bool  device_local =
		    static_cast<bool>(properties.flags & vk::MemoryHeapFlagBits::eDeviceLocal);
		if (device_local) {
			local += properties.size;
		}
		if (!discrete || device_local) {
			budget += CanReportMemoryUsage() ? budgets[heap].budget : properties.size;
			usage += CanReportMemoryUsage() ? budgets[heap].usage : 0;
		}
	}
	if (discrete) {
		return budget - std::min<uint64_t>(budget / 8, 1024ull * 1024 * 1024);
	}
	constexpr uint64_t system_reserve = 8ull * 1024 * 1024 * 1024;
	const auto         available      = budget > usage ? budget - usage : uint64_t {0};
	return std::max(local, available > system_reserve ? available - system_reserve : uint64_t {0});
}

bool GraphicContext::CreateImage(const vk::ImageCreateInfo& image_info, VulkanImage& image) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image != nullptr || image.allocation != nullptr);

	VmaAllocationCreateInfo alloc_info {};
	alloc_info.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

	vk::Image::CType native_image = VK_NULL_HANDLE;
	const auto        result       = static_cast<vk::Result>(
	    vmaCreateImage(allocator, static_cast<const vk::ImageCreateInfo::NativeType*>(image_info),
	                   &alloc_info, &native_image, &image.allocation, nullptr));
	image.image = native_image;
	if (result != vk::Result::eSuccess) {
		LogMemoryBudget();
		return false;
	}

	image.format     = image_info.format;
	image.image_type = image_info.imageType;
	image.extent     = image_info.extent;
	image.layers     = image_info.arrayLayers;
	image.mip_levels = image_info.mipLevels;
	image.samples    = static_cast<uint32_t>(image_info.samples);
	image.usage      = image_info.usage;
	image.flags      = image_info.flags;
	image.state      = {.layout = image_info.initialLayout};
	image.subresource_states.clear();

	return true;
}

void GraphicContext::DeleteImage(VulkanImage& image) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image == nullptr || image.allocation == nullptr);

	vmaDestroyImage(allocator, image.image, image.allocation);
	image.image      = nullptr;
	image.allocation = nullptr;
}

} // namespace Libs::Graphics
