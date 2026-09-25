// Times one vkCreateComputePipelines call for a SPIR-V compute module on the RX 9070 XT and
// prints the driver's pipeline executable statistics. Runs at idle priority, no pipeline cache.
// usage: vk_pipeline_time <file.spv>
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <windows.h>

#include <cstdio>
#include <cstring>
#include <vector>

#define LOAD(name) auto name = reinterpret_cast<PFN_##name>(gipa(instance, #name))
#define LOADD(name) auto name = reinterpret_cast<PFN_##name>(gdpa(device, #name))

int main(int argc, char** argv) {
	if (argc != 2) {
		return 1;
	}
	SetPriorityClass(GetCurrentProcess(), IDLE_PRIORITY_CLASS);
	FILE* f = std::fopen(argv[1], "rb");
	if (f == nullptr) {
		return 2;
	}
	std::vector<uint32_t> code;
	uint32_t              word;
	while (std::fread(&word, 4, 1, f) == 1) {
		code.push_back(word);
	}
	std::fclose(f);

	HMODULE lib  = LoadLibraryA("vulkan-1.dll");
	auto    gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(lib, "vkGetInstanceProcAddr"));
	VkInstance instance = VK_NULL_HANDLE;
	LOAD(vkCreateInstance);
	VkApplicationInfo    app {VK_STRUCTURE_TYPE_APPLICATION_INFO};
	app.apiVersion = VK_API_VERSION_1_3;
	VkInstanceCreateInfo ici {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
	ici.pApplicationInfo = &app;
	if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS) {
		return 3;
	}
	LOAD(vkEnumeratePhysicalDevices);
	LOAD(vkGetPhysicalDeviceProperties);
	LOAD(vkCreateDevice);
	LOAD(vkGetDeviceProcAddr);
	uint32_t count = 0;
	vkEnumeratePhysicalDevices(instance, &count, nullptr);
	std::vector<VkPhysicalDevice> gpus(count);
	vkEnumeratePhysicalDevices(instance, &count, gpus.data());
	VkPhysicalDevice gpu = VK_NULL_HANDLE;
	for (auto candidate: gpus) {
		VkPhysicalDeviceProperties props;
		vkGetPhysicalDeviceProperties(candidate, &props);
		if (std::strstr(props.deviceName, "9070") != nullptr) {
			gpu = candidate;
		}
	}
	if (gpu == VK_NULL_HANDLE) {
		return 4;
	}
	float                   priority = 1.0f;
	VkDeviceQueueCreateInfo qci {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
	qci.queueCount       = 1;
	qci.pQueuePriorities = &priority;
	VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR exec_features {
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
	exec_features.pipelineExecutableInfo = VK_TRUE;
	const char*        extensions[]      = {"VK_KHR_pipeline_executable_properties"};
	VkDeviceCreateInfo dci {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
	dci.pNext                   = &exec_features;
	dci.queueCreateInfoCount    = 1;
	dci.pQueueCreateInfos       = &qci;
	dci.enabledExtensionCount   = 1;
	dci.ppEnabledExtensionNames = extensions;
	VkDevice device             = VK_NULL_HANDLE;
	if (vkCreateDevice(gpu, &dci, nullptr, &device) != VK_SUCCESS) {
		return 5;
	}
	auto gdpa = vkGetDeviceProcAddr;
	LOADD(vkCreateShaderModule);
	LOADD(vkCreateDescriptorSetLayout);
	LOADD(vkCreatePipelineLayout);
	LOADD(vkCreateComputePipelines);
	LOADD(vkGetPipelineExecutablePropertiesKHR);
	LOADD(vkGetPipelineExecutableStatisticsKHR);

	VkShaderModuleCreateInfo smci {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
	smci.codeSize = code.size() * 4;
	smci.pCode    = code.data();
	VkShaderModule module;
	if (vkCreateShaderModule(device, &smci, nullptr, &module) != VK_SUCCESS) {
		return 6;
	}
	VkDescriptorSetLayoutBinding binding {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
	                                      VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
	VkDescriptorSetLayoutCreateInfo dslci {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
	dslci.bindingCount = 1;
	dslci.pBindings    = &binding;
	VkDescriptorSetLayout dsl;
	vkCreateDescriptorSetLayout(device, &dslci, nullptr, &dsl);
	VkPipelineLayoutCreateInfo plci {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
	plci.setLayoutCount = 1;
	plci.pSetLayouts    = &dsl;
	VkPipelineLayout layout;
	vkCreatePipelineLayout(device, &plci, nullptr, &layout);

	VkComputePipelineCreateInfo cpci {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
	cpci.flags        = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
	cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
	cpci.stage.module = module;
	cpci.stage.pName  = "main";
	cpci.layout       = layout;
	VkPipeline    pipeline;
	LARGE_INTEGER freq, t0, t1;
	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&t0);
	const auto result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline);
	QueryPerformanceCounter(&t1);
	const double ms = 1000.0 * static_cast<double>(t1.QuadPart - t0.QuadPart) / freq.QuadPart;
	std::printf("%s words=%zu create_ms=%.1f result=%d", argv[1], code.size(), ms, result);
	if (result == VK_SUCCESS) {
		VkPipelineExecutableInfoKHR info {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR};
		info.pipeline        = pipeline;
		info.executableIndex = 0;
		uint32_t stats       = 0;
		vkGetPipelineExecutableStatisticsKHR(device, &info, &stats, nullptr);
		std::vector<VkPipelineExecutableStatisticKHR> values(
		    stats, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR});
		vkGetPipelineExecutableStatisticsKHR(device, &info, &stats, values.data());
		for (const auto& value: values) {
			std::printf(" | %s=", value.name);
			switch (value.format) {
				case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR:
					std::printf("%u", value.value.b32);
					break;
				case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR:
					std::printf("%lld", static_cast<long long>(value.value.i64));
					break;
				case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR:
					std::printf("%llu", static_cast<unsigned long long>(value.value.u64));
					break;
				default: std::printf("%.2f", value.value.f64); break;
			}
		}
	}
	std::printf("\n");
	std::fflush(stdout);
	// Skip teardown: the process exits right away.
	return 0;
}
