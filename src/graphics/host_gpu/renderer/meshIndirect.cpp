#include "graphics/host_gpu/renderer/meshIndirect.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "gpu_tiler_shaders/mesh_indirect_args_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/gpuZones.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>

namespace Libs::Graphics {

namespace {

// Mirrors the push-constant block of mesh_indirect_args.comp.
struct ShaderParams {
	uint32_t arguments_word;
	uint32_t output_word;
	uint32_t index_bytes;
	uint32_t index_low;
	uint32_t index_high;
	uint32_t index_limit;
	uint32_t primitive_size;
	uint32_t primitive_step;
	uint32_t primitives_per_group;
	uint32_t max_groups;
	uint32_t max_instances;
	uint32_t max_total;
};

} // namespace

MeshIndirectArgs::MeshIndirectArgs(GraphicContext& graphics): m_graphics(graphics) {
	const vk::DescriptorSetLayoutBinding bindings[] {
	    {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	};
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	layout_info.bindingCount = std::size(bindings);
	layout_info.pBindings    = bindings;
	RequireVulkanSuccess(
	    graphics.device.createDescriptorSetLayout(&layout_info, nullptr, &m_descriptor_layout),
	    "create mesh indirect descriptor layout");

	const vk::PushConstantRange push_range {vk::ShaderStageFlagBits::eCompute, 0,
	                                        sizeof(ShaderParams)};
	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = 1;
	pipeline_layout_info.pSetLayouts            = &m_descriptor_layout;
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push_range;
	RequireVulkanSuccess(graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
	                                                          &m_pipeline_layout),
	                     "create mesh indirect pipeline layout");

	const auto module = CompileSPV(MESH_INDIRECT_ARGS_SPV, graphics.device);
	vk::PipelineShaderStageCreateInfo stage {};
	stage.stage  = vk::ShaderStageFlagBits::eCompute;
	stage.module = module;
	stage.pName  = "main";
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage  = stage;
	pipeline_info.layout = m_pipeline_layout;
	const auto result =
	    graphics.device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &m_pipeline);
	graphics.device.destroyShaderModule(module, nullptr);
	RequireVulkanSuccess(result, "create mesh indirect pipeline");
	SetVulkanObjectNameF(graphics.device, m_pipeline, "Kyty.MeshIndirectArgs");
}

MeshIndirectArgs::~MeshIndirectArgs() {
	if (m_pipeline != nullptr) {
		m_graphics.device.destroyPipeline(m_pipeline, nullptr);
	}
	if (m_pipeline_layout != nullptr) {
		m_graphics.device.destroyPipelineLayout(m_pipeline_layout, nullptr);
	}
	if (m_descriptor_layout != nullptr) {
		m_graphics.device.destroyDescriptorSetLayout(m_descriptor_layout, nullptr);
	}
}

void MeshIndirectArgs::Record(vk::CommandBuffer command, const Buffer& arguments,
                              uint64_t arguments_offset, const Buffer& output,
                              uint64_t output_offset, const Params& params) {
	GpuZones::Mark(command, DrainStats::Zone::MeshArgs);
	const auto alignment = m_graphics.StorageMinAlignment();
	EXIT_IF(arguments_offset % sizeof(uint32_t) != 0 || output_offset % sizeof(uint32_t) != 0 ||
	        params.primitive_size == 0 || params.primitive_step == 0 ||
	        params.primitives_per_group == 0);
	const auto arguments_binding = Common::AlignDown(arguments_offset, alignment);
	const auto output_binding    = Common::AlignDown(output_offset, alignment);
	const auto arguments_size    = arguments_offset - arguments_binding + ArgumentsSize;
	const auto output_size       = output_offset - output_binding + OutputSize;
	EXIT_IF(arguments_binding > arguments.Size() ||
	        arguments_size > arguments.Size() - arguments_binding ||
	        output_binding > output.Size() || output_size > output.Size() - output_binding);

	// The game's dispatch wrote the arguments earlier in this queue, and an earlier draw may
	// still read a record slot this ring is reusing.
	vk::MemoryBarrier2 before {};
	before.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	before.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
	before.dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	before.dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &before;
	command.pipelineBarrier2(dependency);

	const vk::DescriptorBufferInfo infos[] {
	    {arguments.Handle(), arguments_binding, arguments_size},
	    {output.Handle(), output_binding, output_size},
	};
	std::array<vk::WriteDescriptorSet, 2> writes {};
	for (uint32_t index = 0; index < writes.size(); ++index) {
		writes[index].dstBinding      = index;
		writes[index].descriptorCount = 1;
		writes[index].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[index].pBufferInfo     = &infos[index];
	}
	const ShaderParams shader_params {
	    .arguments_word =
	        static_cast<uint32_t>((arguments_offset - arguments_binding) / sizeof(uint32_t)),
	    .output_word = static_cast<uint32_t>((output_offset - output_binding) / sizeof(uint32_t)),
	    .index_bytes = params.index_bytes,
	    .index_low   = static_cast<uint32_t>(params.index_address),
	    .index_high  = static_cast<uint32_t>(params.index_address >> 32u),
	    .index_limit = params.index_limit,
	    .primitive_size       = params.primitive_size,
	    .primitive_step       = params.primitive_step,
	    .primitives_per_group = params.primitives_per_group,
	    .max_groups           = params.max_groups,
	    .max_instances        = params.max_instances,
	    .max_total            = params.max_total,
	};
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_pipeline_layout, 0, writes);
	command.pushConstants(m_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0,
	                      sizeof(shader_params), &shader_params);
	command.dispatch(1, 1, 1);

	vk::MemoryBarrier2 after {};
	after.srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	after.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
	after.dstStageMask  = vk::PipelineStageFlagBits2::eDrawIndirect |
	                     vk::PipelineStageFlagBits2::eMeshShaderEXT;
	after.dstAccessMask = vk::AccessFlagBits2::eIndirectCommandRead |
	                      vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eMemoryRead;
	dependency.pMemoryBarriers = &after;
	command.pipelineBarrier2(dependency);
}

} // namespace Libs::Graphics
