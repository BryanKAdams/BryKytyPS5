#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineLibrary.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/rectListShader.h"
#include "graphics/shader/shader.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <future>
#include <limits>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

namespace Libs::Graphics {

// IDK: maybe we can remove it?
constexpr uint8_t kTemporaryVertexAttribFormat113 =
    static_cast<uint8_t>(Prospero::VertexAttribFormat::k16_16SInt);
constexpr uint32_t kTemporaryPs5BufferFormat121 = 121u;

static bool NarrowInputFormat(vk::Format& format, uint32_t& size, uint32_t used_components) {
	if (used_components == 0 || used_components >= size) {
		return false;
	}

	switch (format) {
		case vk::Format::eR32G32B32A32Sfloat:
			switch (used_components) {
				case 1: format = vk::Format::eR32Sfloat; break;
				case 2: format = vk::Format::eR32G32Sfloat; break;
				case 3: format = vk::Format::eR32G32B32Sfloat; break;
				default: return false;
			}
			size = used_components;
			return true;
		case vk::Format::eR32G32B32Sfloat:
			switch (used_components) {
				case 1: format = vk::Format::eR32Sfloat; break;
				case 2: format = vk::Format::eR32G32Sfloat; break;
				default: return false;
			}
			size = used_components;
			return true;
		case vk::Format::eR16G16B16A16Sfloat:
			switch (used_components) {
				case 1: format = vk::Format::eR16Sfloat; break;
				case 2: format = vk::Format::eR16G16Sfloat; break;
				default: return false;
			}
			size = used_components;
			return true;
		case vk::Format::eR8G8B8A8Unorm:
			switch (used_components) {
				case 1: format = vk::Format::eR8Unorm; break;
				case 2: format = vk::Format::eR8G8Unorm; break;
				default: return false;
			}
			size = used_components;
			return true;
		case vk::Format::eR8G8B8A8Snorm:
			if (used_components != 2) {
				return false;
			}
			format = vk::Format::eR8G8Snorm;
			size   = 2;
			return true;
		case vk::Format::eR8G8B8A8Uint:
			switch (used_components) {
				case 1: format = vk::Format::eR8Uint; break;
				case 2: format = vk::Format::eR8G8Uint; break;
				default: return false;
			}
			size = used_components;
			return true;
		default: break;
	}

	return false;
}

static void GetInputFormat(const ShaderBufferResource& res, vk::Format& format, uint32_t& size,
                           uint32_t used_components) {
	const auto fmt        = res.Format();
	const auto raw_format = res.RawFormat();
	if (raw_format == kTemporaryVertexAttribFormat113) {
		static bool logged_113 = false;
		if (!logged_113) {
			LOGF("InputFormat: temporary: accepting invalid PS5 buffer format 113 as "
			     "vk::Format::eR32G32B32A32Sfloat\n");
			logged_113 = true;
		}
		format = vk::Format::eR32G32B32A32Sfloat;
		size   = 4;
		if (NarrowInputFormat(format, size, used_components)) {
			LOGF("InputFormat: narrowing fmt=%u to %s for used_components=%u\n", raw_format,
			     vk::to_string(format).c_str(), used_components);
		}
		return;
	}
	if (raw_format == kTemporaryPs5BufferFormat121) {
		static bool logged_121 = false;
		if (!logged_121) {
			LOGF("InputFormat: accepting PS5 buffer format 121 as vk::Format::eR16G16Sfloat\n");
			logged_121 = true;
		}
		format = vk::Format::eR16G16Sfloat;
		size   = 2;
		return;
	}

	format = VulkanFormat(fmt);
	size   = ShaderRecompiler::Format::GetFormatInfo(fmt).component_count;
	if (format == vk::Format::eUndefined || size == 0) {
		EXIT("unknown vertex format: fmt = %u\n", raw_format);
	}

	if (NarrowInputFormat(format, size, used_components)) {
		static std::atomic<uint64_t> log_count = 0;
		auto                         log_id    = log_count.fetch_add(1);
		if (log_id < 32) {
			LOGF("VertexInput: narrowed vertex format to %" PRIu32
			     " component(s) for shader fetch\n",
			     used_components);
		}
	}
}

static vk::BlendFactor GetBlendFactor(uint32_t factor) {
	switch (static_cast<Prospero::BlendFactor>(factor)) {
		case Prospero::BlendFactor::kZero: return vk::BlendFactor::eZero;
		case Prospero::BlendFactor::kOne: return vk::BlendFactor::eOne;
		case Prospero::BlendFactor::kSrcColor: return vk::BlendFactor::eSrcColor;
		case Prospero::BlendFactor::kOneMinusSrcColor: return vk::BlendFactor::eOneMinusSrcColor;
		case Prospero::BlendFactor::kSrcAlpha: return vk::BlendFactor::eSrcAlpha;
		case Prospero::BlendFactor::kOneMinusSrcAlpha: return vk::BlendFactor::eOneMinusSrcAlpha;
		case Prospero::BlendFactor::kDstAlpha: return vk::BlendFactor::eDstAlpha;
		case Prospero::BlendFactor::kOneMinusDstAlpha: return vk::BlendFactor::eOneMinusDstAlpha;
		case Prospero::BlendFactor::kDstColor: return vk::BlendFactor::eDstColor;
		case Prospero::BlendFactor::kOneMinusDstColor: return vk::BlendFactor::eOneMinusDstColor;
		case Prospero::BlendFactor::kSrcAlphaSaturate: return vk::BlendFactor::eSrcAlphaSaturate;
		case Prospero::BlendFactor::kConstantColor: return vk::BlendFactor::eConstantColor;
		case Prospero::BlendFactor::kOneMinusConstantColor:
			return vk::BlendFactor::eOneMinusConstantColor;
		case Prospero::BlendFactor::kSrc1Color: return vk::BlendFactor::eSrc1Color;
		case Prospero::BlendFactor::kOneMinusSrc1Color: return vk::BlendFactor::eOneMinusSrc1Color;
		case Prospero::BlendFactor::kSrc1Alpha: return vk::BlendFactor::eSrc1Alpha;
		case Prospero::BlendFactor::kOneMinusSrc1Alpha: return vk::BlendFactor::eOneMinusSrc1Alpha;
		case Prospero::BlendFactor::kConstantAlpha: return vk::BlendFactor::eConstantAlpha;
		case Prospero::BlendFactor::kOneMinusConstantAlpha:
			return vk::BlendFactor::eOneMinusConstantAlpha;
		default: EXIT("unknown factor: %u\n", factor);
	}
	return vk::BlendFactor::eZero;
}

static vk::BlendOp GetBlendOp(uint32_t op) {
	switch (static_cast<Prospero::BlendOp>(op)) {
		case Prospero::BlendOp::kAdd: return vk::BlendOp::eAdd;
		case Prospero::BlendOp::kSubtract: return vk::BlendOp::eSubtract;
		case Prospero::BlendOp::kMin: return vk::BlendOp::eMin;
		case Prospero::BlendOp::kMax: return vk::BlendOp::eMax;
		case Prospero::BlendOp::kReverseSubtract: return vk::BlendOp::eReverseSubtract;
		default: EXIT("unknown op: %u\n", op);
	}
	return vk::BlendOp::eAdd;
}

static void AddLayoutBindings(std::vector<vk::DescriptorSetLayoutBinding>& descriptor_bindings,
                              const ShaderRecompiler::IR::CompiledShaderInfo& program,
                              vk::ShaderStageFlagBits              stage) {
	for (const auto& binding: program.bindings.descriptors) {
		descriptor_bindings.push_back(
		    {ShaderRecompiler::IR::NativeBinding(program.stage, binding.kind),
		     NativeDescriptorType(binding.kind), NativeDescriptorCount(binding), stage, nullptr});
	}
}

[[nodiscard]] static bool FitsPushDescriptors(
    const GraphicContext& graphics, std::span<const vk::DescriptorSetLayoutBinding> bindings) {
	uint32_t descriptor_count = 0;
	for (const auto& binding: bindings) {
		descriptor_count += binding.descriptorCount;
	}
	return descriptor_count <= graphics.max_push_descriptors;
}

[[nodiscard]] static vk::DescriptorSetLayout
CreateSetLayout(GraphicContext& graphics, std::span<const vk::DescriptorSetLayoutBinding> bindings,
                bool push) {
	vk::DescriptorSetLayoutCreateInfo create {};
	create.flags        = push ? vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR
	                           : vk::DescriptorSetLayoutCreateFlags {};
	create.bindingCount = static_cast<uint32_t>(bindings.size());
	create.pBindings    = bindings.data();
	vk::DescriptorSetLayout layout = nullptr;
	EXIT_IF(graphics.device.createDescriptorSetLayout(&create, nullptr, &layout) !=
	        vk::Result::eSuccess);
	return layout;
}

static void CreateDescriptorLayout(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                                   std::span<const vk::DescriptorSetLayoutBinding> bindings) {
	pipeline.uses_push_descriptors = FitsPushDescriptors(graphics, bindings);
	pipeline.descriptor_set_layout =
	    CreateSetLayout(graphics, bindings, pipeline.uses_push_descriptors);
}

namespace {

// Every create-info structure of one graphics pipeline. The monolithic path passes them all to one
// vkCreateGraphicsPipelines call; the pipeline-library path splits them between the four library
// parts. The structures point into each other, so the state is built in place and never moved.
struct GraphicsPipelineState {
	explicit GraphicsPipelineState(vk::Device device): device(device) {}
	~GraphicsPipelineState() {
		for (const auto module: owned_modules) {
			if (module != nullptr) {
				device.destroyShaderModule(module, nullptr);
			}
		}
	}
	KYTY_CLASS_NO_COPY(GraphicsPipelineState);

	vk::Device device;
	// The RectList tessellation shaders built for this pipeline.
	std::array<vk::ShaderModule, 2> owned_modules {};
	bool                            mesh       = false;
	bool                            rect_list  = false;
	bool                            with_depth = false;

	std::array<vk::PipelineShaderStageCreateInfo, 4> stages {};
	uint32_t                                         stage_count = 0;
	// The fragment stage, when there is one, comes last.
	uint32_t fragment_stage_count = 0;

	std::array<vk::VertexInputAttributeDescription, ShaderVertexInputInfo::RES_MAX> input_attr {};
	std::array<vk::VertexInputBindingDescription, ShaderVertexInputInfo::RES_MAX>   input_desc {};
	vk::PipelineVertexInputStateCreateInfo            vertex_input {};
	vk::PipelineInputAssemblyStateCreateInfo          input_assembly {};
	vk::PipelineTessellationStateCreateInfo           tessellation {};
	bool                                              uses_tessellation = false;
	vk::PipelineViewportDepthClipControlCreateInfoEXT depth_clip_control {};
	vk::PipelineViewportStateCreateInfo               viewport {};
	vk::PipelineRasterizationDepthClipStateCreateInfoEXT       depth_clip {};
	vk::PipelineRasterizationProvokingVertexStateCreateInfoEXT provoking_vertex {};
	vk::PipelineRasterizationStateCreateInfo                   rasterizer {};
	vk::PipelineMultisampleStateCreateInfo                     multisampling {};
	vk::PipelineDepthStencilStateCreateInfo                    depth_stencil {};
	std::array<vk::PipelineColorBlendAttachmentState, RENDER_COLOR_ATTACHMENTS_MAX> blend {};
	std::array<vk::Bool32, RENDER_COLOR_ATTACHMENTS_MAX> color_write_enable {};
	vk::PipelineColorWriteCreateInfoEXT                  color_write {};
	vk::PipelineColorBlendStateCreateInfo                color_blending {};
	// Dynamic states; color write enable, a fragment output state, is last when present.
	std::vector<vk::DynamicState>      dynamic_states;
	uint32_t                           shared_dynamic_state_count = 0;
	vk::PipelineDynamicStateCreateInfo dynamic_state {};
	vk::PipelineRenderingCreateInfo    rendering {};
	vk::PipelineCreateFlags            flags {};

	// Set 0 holds the vertex-side stages' descriptors and set 1 the pixel shader's (the
	// recompiler decorates pixel shader resources with DescriptorSet 1).
	std::vector<vk::DescriptorSetLayoutBinding> vertex_bindings;
	std::vector<vk::DescriptorSetLayoutBinding> pixel_bindings;
	vk::ShaderStageFlags                        graphics_stages;
	// A pipeline layout can hold one push descriptor set: the pixel set pushes when it fits,
	// otherwise the vertex set may.
	bool pixel_uses_push  = false;
	bool vertex_uses_push = false;
};

// A library part serves pipelines with different vertex-side stages, and every part of one pipeline
// must use an identical layout, so library pipelines use one push constant range for all stages.
constexpr vk::ShaderStageFlags LibraryPushConstantStages =
    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eTessellationControl |
    vk::ShaderStageFlagBits::eTessellationEvaluation | vk::ShaderStageFlagBits::eFragment;

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void BuildGraphicsPipelineState(GraphicsPipelineState& state, GraphicContext& graphics,
                                const PipelineRenderingState&          rendering,
                                const PipelineVertexInputState&        vertex_input,
                                std::span<const ShaderVertexInputInfo> vertex_info,
                                const ShaderPixelInputInfo*            ps_input_info,
                                const PipelineCache::GraphicsPrograms& programs,
                                const PipelineStaticParameters&        static_params) {
	const auto& vs_input_info  = vertex_info.front();
	const auto& vertex_program = programs.vertex[0];
	const auto& pixel_program  = programs.pixel;
	const bool  tessellation   = vertex_info.size() == 3;
	const bool  ps_active      = ps_input_info != nullptr;
	EXIT_IF(!vertex_program || (ps_active && !pixel_program));
	state.with_depth = rendering.depth_format != vk::Format::eUndefined ||
	                   rendering.stencil_format != vk::Format::eUndefined;
	EXIT_IF(!vs_input_info.stage);
	state.mesh = vs_input_info.stage.program->stage == ShaderType::Mesh;
	EXIT_NOT_IMPLEMENTED(state.mesh && !graphics.mesh_shader_enabled);
	state.rect_list = !state.mesh && !tessellation &&
	                  static_params.topology == vk::PrimitiveTopology::ePatchList;

	vk::ShaderModule tess_control_shader_module = nullptr;
	vk::ShaderModule tess_eval_shader_module    = nullptr;

	if (state.rect_list) {
		const auto shaders =
		    BuildRectListShaders(vs_input_info, ps_active ? ps_input_info : nullptr);
		tess_control_shader_module = CompileSPV(shaders.control, graphics.device);
		state.owned_modules[0]     = tess_control_shader_module;
		if (graphics_debug_dump_enabled()) {
			LOGF("PipelineTrace: vkCreateShaderModule RectList TCS done module=%p\n",
			     static_cast<void*>(tess_control_shader_module));
		}

		tess_eval_shader_module = CompileSPV(shaders.evaluation, graphics.device);
		state.owned_modules[1]  = tess_eval_shader_module;
		if (graphics_debug_dump_enabled()) {
			LOGF("PipelineTrace: vkCreateShaderModule RectList TES done module=%p\n",
			     static_cast<void*>(tess_eval_shader_module));
		}
	}

	EXIT_NOT_IMPLEMENTED(state.rect_list && (tess_control_shader_module == nullptr ||
	                                         tess_eval_shader_module == nullptr));

	auto& shader_stages = state.stages;
	auto& stage_count   = state.stage_count;
	for (uint32_t i = 0; i < vertex_info.size(); i++) {
		shader_stages[stage_count++] = {.stage  = NativeShaderStage(vertex_info[i].logical_stage),
		                                .module = programs.vertex[i].module,
		                                .pName  = "main"};
	}
	if (state.rect_list) {
		shader_stages[stage_count++] = {.stage  = vk::ShaderStageFlagBits::eTessellationControl,
		                                .module = tess_control_shader_module,
		                                .pName  = "main"};
		shader_stages[stage_count++] = {.stage  = vk::ShaderStageFlagBits::eTessellationEvaluation,
		                                .module = tess_eval_shader_module,
		                                .pName  = "main"};
	}
	if (ps_active) {
		shader_stages[stage_count++] = {.stage  = vk::ShaderStageFlagBits::eFragment,
		                                .module = pixel_program.module,
		                                .pName  = "main"};
		state.fragment_stage_count   = 1;
	}

	auto& input_attr = state.input_attr;
	auto& input_desc = state.input_desc;

	for (uint32_t binding = 0; binding < vertex_input.binding_count; binding++) {
		input_desc[binding].binding   = binding;
		input_desc[binding].stride    = vertex_input.bindings[binding].stride;
		input_desc[binding].inputRate = vertex_input.bindings[binding].instance
		                                    ? vk::VertexInputRate::eInstance
		                                    : vk::VertexInputRate::eVertex;
	}
	for (uint32_t index = 0; index < vertex_input.attribute_count; index++) {
		input_attr[index].binding  = vertex_input.attributes[index].binding;
		input_attr[index].location = index;
		input_attr[index].offset   = vertex_input.attributes[index].offset;

		uint32_t   attr_size     = 4;
		const auto registers_num = vs_input_info.resources_dst[index].registers_num;
		const auto compiled_components =
		    vs_input_info.stage.program->info.vertex_fetch_components[index];
		const auto used_components =
		    compiled_components > 0 ? static_cast<int>(compiled_components) : registers_num;
		GetInputFormat(vs_input_info.resources[index], input_attr[index].format, attr_size,
		               static_cast<uint32_t>(used_components));

		if (graphics_debug_dump_enabled()) {
			static std::atomic_uint log_count = 0;
			const auto              log_id    = log_count.fetch_add(1, std::memory_order_relaxed);
			if (log_id < 128) {
				LOGF("VertexInputState[%u]: attr=%u binding=%u offset=%u stride=%u fmt=%d "
				     "src_fmt=%u dst=v%u regs=%u"
				     " fetched_components=%u attr_size=%u swizzle=%u,%u,%u,%u\n",
				     log_id, index, input_attr[index].binding, input_attr[index].offset,
				     input_desc[input_attr[index].binding].stride,
				     static_cast<int>(input_attr[index].format),
				     static_cast<uint32_t>(vs_input_info.resources[index].Format()),
				     static_cast<uint32_t>(vs_input_info.resources_dst[index].register_start),
				     static_cast<uint32_t>(registers_num), static_cast<uint32_t>(used_components),
				     attr_size, static_cast<uint32_t>(vs_input_info.resources[index].DstSelX()),
				     static_cast<uint32_t>(vs_input_info.resources[index].DstSelY()),
				     static_cast<uint32_t>(vs_input_info.resources[index].DstSelZ()),
				     static_cast<uint32_t>(vs_input_info.resources[index].DstSelW()));
			}
		}

		if (vs_input_info.resources[index].OutOfBounds() != 0) {
			static bool logged = false;
			if (!logged) {
				LOGF("VertexInput: temporary: accepting PS5 out-of-bounds behavior %" PRIu8 "\n",
				     vs_input_info.resources[index].OutOfBounds());
				logged = true;
			}
		}

		EXIT_NOT_IMPLEMENTED(vs_input_info.resources[index].AddTid());
		EXIT_NOT_IMPLEMENTED(vs_input_info.resources[index].SwizzleEnabled());

		EXIT_IF(registers_num < 1 || registers_num > 4);
	}

	auto& vertex_input_info                           = state.vertex_input;
	vertex_input_info.vertexBindingDescriptionCount   = vertex_input.binding_count;
	vertex_input_info.pVertexBindingDescriptions      = input_desc.data();
	vertex_input_info.vertexAttributeDescriptionCount = vertex_input.attribute_count;
	vertex_input_info.pVertexAttributeDescriptions    = input_attr.data();

	auto& input_assembly    = state.input_assembly;
	input_assembly.topology = static_params.topology;
	input_assembly.primitiveRestartEnable =
	    static_params.primitive_restart_enable ? VK_TRUE : VK_FALSE;

	auto& depth_clip_control = state.depth_clip_control;
	depth_clip_control.negativeOneToOne = (static_params.negative_one_to_one ? VK_TRUE : VK_FALSE);

	auto& viewport_state = state.viewport;
	viewport_state.pNext = &depth_clip_control;

	vk::CullModeFlags cull_mode = vk::CullModeFlagBits::eNone;
	if (static_params.cull_back) {
		cull_mode |= vk::CullModeFlagBits::eBack;
	}
	if (static_params.cull_front) {
		cull_mode |= vk::CullModeFlagBits::eFront;
	}

	vk::FrontFace front_face =
	    (static_params.face ? vk::FrontFace::eClockwise : vk::FrontFace::eCounterClockwise);

	auto& clip_ext           = state.depth_clip;
	clip_ext.depthClipEnable = static_params.depth_clip_enable ? VK_TRUE : VK_FALSE;

	auto& rasterizer = state.rasterizer;
	// MoltenVK lacks VK_EXT_depth_clip_enable; omit the depth-clip struct on macOS and accept
	// Vulkan's default depth clipping (enabled) instead of the PS5's clamp behavior.
#if !defined(__APPLE__)
	// The DB clamps depth to the viewport range after polygon offset is applied.
	rasterizer.depthClampEnable = VK_TRUE;
	rasterizer.pNext = &clip_ext;
#endif
	auto& provoking_vertex = state.provoking_vertex;
	EXIT_NOT_IMPLEMENTED(static_params.provoking_vtx_last &&
	                     !graphics.provoking_vertex_last_enabled);
	if (graphics.provoking_vertex_last_enabled) {
		provoking_vertex.provokingVertexMode = static_params.provoking_vtx_last
		    ? vk::ProvokingVertexModeEXT::eLastVertex : vk::ProvokingVertexModeEXT::eFirstVertex;
		provoking_vertex.pNext = rasterizer.pNext;
		rasterizer.pNext = &provoking_vertex;
	}
	rasterizer.cullMode  = cull_mode;
	rasterizer.frontFace = front_face;
	rasterizer.polygonMode = static_params.polygon_mode;
	rasterizer.lineWidth = 1.0f;

	auto& multisampling                = state.multisampling;
	multisampling.sampleShadingEnable  = static_params.sample_shading_enable ? VK_TRUE : VK_FALSE;
	multisampling.rasterizationSamples = vulkan_sample_count(static_params.samples);
	multisampling.minSampleShading     = 1.0f;

	auto& color_blend_attachment = state.blend;
	for (uint32_t i = 0; i < rendering.color_count; i++) {
		EXIT_NOT_IMPLEMENTED((static_params.color_mask[i] & ~0x0fu) != 0);
		color_blend_attachment[i].colorWriteMask =
		    vk::ColorComponentFlags {static_params.color_mask[i]};
		color_blend_attachment[i].blendEnable = static_params.blend_enable[i] ? VK_TRUE : VK_FALSE;
		color_blend_attachment[i].srcColorBlendFactor =
		    GetBlendFactor(static_params.color_srcblend[i]);
		color_blend_attachment[i].dstColorBlendFactor =
		    GetBlendFactor(static_params.color_destblend[i]);
		color_blend_attachment[i].colorBlendOp = GetBlendOp(static_params.color_comb_fcn[i]);
		color_blend_attachment[i].srcAlphaBlendFactor =
		    (static_params.separate_alpha_blend[i] ? GetBlendFactor(static_params.alpha_srcblend[i])
		                                           : color_blend_attachment[i].srcColorBlendFactor);
		color_blend_attachment[i].dstAlphaBlendFactor =
		    (static_params.separate_alpha_blend[i]
		         ? GetBlendFactor(static_params.alpha_destblend[i])
		         : color_blend_attachment[i].dstColorBlendFactor);
		color_blend_attachment[i].alphaBlendOp =
		    (static_params.separate_alpha_blend[i] ? GetBlendOp(static_params.alpha_comb_fcn[i])
		                                           : color_blend_attachment[i].colorBlendOp);
	}

	for (uint32_t i = 0; i < rendering.color_count; i++) {
		state.color_write_enable[i] = VK_TRUE;
	}

	auto& color_write              = state.color_write;
	color_write.attachmentCount    = rendering.color_count;
	color_write.pColorWriteEnables = state.color_write_enable.data();

	auto& color_blending = state.color_blending;
	// MoltenVK lacks VK_EXT_color_write_enable; drop the dynamic color-write struct on macOS
	// and rely on each attachment's static colorWriteMask (all channels enabled by default).
#if !defined(__APPLE__)
	color_blending.pNext = &color_write;
#endif
	color_blending.logicOp         = vk::LogicOp::eCopy;
	color_blending.attachmentCount = rendering.color_count;
	color_blending.pAttachments    = color_blend_attachment.data();

	vk::ShaderStageFlags graphics_stages = vk::ShaderStageFlagBits::eFragment;
	for (const auto& stage: vertex_info) {
		const auto native_stage = NativeShaderStage(stage.logical_stage);
		AddLayoutBindings(state.vertex_bindings, *stage.stage.program, native_stage);
		graphics_stages |= native_stage;
	}
	if (ps_active) {
		EXIT_IF(!ps_input_info->stage);
		AddLayoutBindings(state.pixel_bindings, *ps_input_info->stage.program,
		                  vk::ShaderStageFlagBits::eFragment);
	}
	state.graphics_stages = graphics_stages;
	state.pixel_uses_push =
	    !state.pixel_bindings.empty() && FitsPushDescriptors(graphics, state.pixel_bindings);
	state.vertex_uses_push = !state.pixel_uses_push && !state.vertex_bindings.empty() &&
	                         FitsPushDescriptors(graphics, state.vertex_bindings);

	auto& depth_stencil_info = state.depth_stencil;
	depth_stencil_info.depthBoundsTestEnable =
#if defined(__APPLE__)
	    VK_FALSE; // MoltenVK lacks the depthBounds feature; depth-bounds testing is disabled
#else
	    (static_params.depth_bounds_test_enable ? VK_TRUE : VK_FALSE);
#endif
	depth_stencil_info.minDepthBounds    = static_params.depth_min_bounds;
	depth_stencil_info.maxDepthBounds    = static_params.depth_max_bounds;

	auto& dynamic_states = state.dynamic_states;
	dynamic_states       = {
        vk::DynamicState::eViewportWithCount,
        vk::DynamicState::eScissorWithCount,
        vk::DynamicState::eLineWidth,
        vk::DynamicState::eDepthTestEnable,
        vk::DynamicState::eDepthWriteEnable,
        vk::DynamicState::eDepthCompareOp,
        vk::DynamicState::eDepthBiasEnable,
        vk::DynamicState::eDepthBias,
        vk::DynamicState::eStencilTestEnable,
        vk::DynamicState::eStencilOp,
        vk::DynamicState::eStencilCompareMask,
        vk::DynamicState::eStencilReference,
        vk::DynamicState::eStencilWriteMask,
        vk::DynamicState::eBlendConstants,
    };
	if (graphics.attachment_feedback_loop_dynamic_enabled) {
		dynamic_states.push_back(vk::DynamicState::eAttachmentFeedbackLoopEnableEXT);
	}
	state.shared_dynamic_state_count = static_cast<uint32_t>(dynamic_states.size());
#if !defined(__APPLE__)
	if (rendering.color_count != 0) {
		dynamic_states.push_back(vk::DynamicState::eColorWriteEnableEXT);
	}
#endif

	auto& dynamic_state             = state.dynamic_state;
	dynamic_state.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
	dynamic_state.pDynamicStates    = dynamic_states.data();

	state.flags                             = static_params.attachment_feedback_loop_flags;
	state.rendering.colorAttachmentCount    = rendering.color_count;
	state.rendering.pColorAttachmentFormats = rendering.color_formats.data();
	state.rendering.depthAttachmentFormat   = rendering.depth_format;
	state.rendering.stencilAttachmentFormat = rendering.stencil_format;
	state.tessellation.patchControlPoints =
	    tessellation ? vs_input_info.tess.input_control_points : 3u;
	state.uses_tessellation = state.rect_list || tessellation;
}

vk::PipelineLayout CreatePipelineLayout(GraphicContext&                        graphics,
                                        std::span<const vk::DescriptorSetLayout> set_layouts,
                                        vk::ShaderStageFlags                     push_stages) {
	const vk::PushConstantRange push_constants {push_stages, 0,
	                                            ShaderRecompiler::IR::NativePushConstantSize};
	vk::PipelineLayoutCreateInfo info {};
	info.setLayoutCount         = static_cast<uint32_t>(set_layouts.size());
	info.pSetLayouts            = set_layouts.data();
	info.pushConstantRangeCount = 1;
	info.pPushConstantRanges    = &push_constants;
	vk::PipelineLayout layout   = nullptr;
	EXIT_NOT_IMPLEMENTED(graphics.device.createPipelineLayout(&info, nullptr, &layout) !=
	                     vk::Result::eSuccess);
	return layout;
}

// The pipeline's own two descriptor sets and its pipeline layout.
void CreateGraphicsLayouts(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                           const GraphicsPipelineState& state, vk::ShaderStageFlags push_stages) {
	pipeline.uses_push_descriptors = state.vertex_uses_push;
	pipeline.pixel_uses_push       = state.pixel_uses_push;
	pipeline.descriptor_set_layout =
	    CreateSetLayout(graphics, state.vertex_bindings, state.vertex_uses_push);
	pipeline.pixel_set_layout = CreateSetLayout(graphics, state.pixel_bindings, state.pixel_uses_push);
	const std::array set_layouts {pipeline.descriptor_set_layout, pipeline.pixel_set_layout};
	EXIT_IF(pipeline.pipeline_layout != nullptr);
	pipeline.pipeline_layout = CreatePipelineLayout(graphics, set_layouts, push_stages);
}

void CreateMonolithicPipeline(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                              const GraphicsPipelineState& state, vk::PipelineCache driver_cache) {
	CreateGraphicsLayouts(graphics, pipeline, state, state.graphics_stages);

	vk::GraphicsPipelineCreateInfo pipeline_info {};
	pipeline_info.flags               = state.flags;
	pipeline_info.pNext               = &state.rendering;
	pipeline_info.stageCount          = state.stage_count;
	pipeline_info.pStages             = state.stages.data();
	pipeline_info.pVertexInputState   = state.mesh ? nullptr : &state.vertex_input;
	pipeline_info.pInputAssemblyState = state.mesh ? nullptr : &state.input_assembly;
	pipeline_info.pTessellationState  = state.uses_tessellation ? &state.tessellation : nullptr;
	pipeline_info.pViewportState      = &state.viewport;
	pipeline_info.pRasterizationState = &state.rasterizer;
	pipeline_info.pMultisampleState   = &state.multisampling;
	pipeline_info.pDepthStencilState  = (state.with_depth ? &state.depth_stencil : nullptr);
	pipeline_info.pColorBlendState    = &state.color_blending;
	pipeline_info.pDynamicState       = &state.dynamic_state;
	pipeline_info.layout              = pipeline.pipeline_layout;
	pipeline_info.basePipelineIndex   = -1;

	EXIT_IF(pipeline.pipeline != nullptr);
	const auto result = graphics.device.createGraphicsPipelines(driver_cache, 1, &pipeline_info,
	                                                            nullptr, &pipeline.pipeline);
	if (graphics_debug_dump_enabled()) {
		LOGF("PipelineTrace: vkCreateGraphicsPipelines done result=%s pipeline=%p\n",
		     vk::to_string(result).c_str(), static_cast<void*>(pipeline.pipeline));
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	EXIT_NOT_IMPLEMENTED(pipeline.pipeline == nullptr);
}

template <typename T>
void AppendKey(std::string& key, const T& value) {
	static_assert(std::is_trivially_copyable_v<T>);
	key.append(reinterpret_cast<const char*>(&value), sizeof(value));
}

void AppendBindingsKey(std::string& key, std::span<const vk::DescriptorSetLayoutBinding> bindings,
                       bool push) {
	AppendKey(key, static_cast<uint32_t>(bindings.size()));
	for (const auto& binding: bindings) {
		AppendKey(key, binding.binding);
		AppendKey(key, binding.descriptorType);
		AppendKey(key, binding.descriptorCount);
		AppendKey(key, static_cast<uint32_t>(binding.stageFlags));
	}
	AppendKey(key, push);
}

// Creates one library part from the parts of `info` that belong to it.
vk::Pipeline CreateLibraryPart(GraphicContext& graphics, vk::GraphicsPipelineLibraryFlagsEXT part,
                               vk::GraphicsPipelineCreateInfo         info,
                               const vk::PipelineRenderingCreateInfo& rendering,
                               vk::PipelineCache                      driver_cache) {
	vk::GraphicsPipelineLibraryCreateInfoEXT library {};
	library.pNext = const_cast<vk::PipelineRenderingCreateInfo*>(&rendering);
	library.flags = part;
	info.pNext    = &library;
	info.flags |= vk::PipelineCreateFlagBits::eLibraryKHR |
	              vk::PipelineCreateFlagBits::eRetainLinkTimeOptimizationInfoEXT;
	info.basePipelineIndex = -1;
	vk::Pipeline pipeline  = nullptr;
	EXIT_NOT_IMPLEMENTED(graphics.device.createGraphicsPipelines(driver_cache, 1, &info, nullptr,
	                                                             &pipeline) != vk::Result::eSuccess);
	return pipeline;
}

// Builds the pipeline from graphics-pipeline-library parts, compiling the parts no earlier pipeline
// built, and fast-links them. Returns a bit per compiled part.
//
// Every part uses the pipeline's full layout. Layouts with independent sets would let a shader
// part ignore the other stage's descriptor set, but on AMD's 26.6 driver any pipeline with such a
// layout (even a monolithic one) lost the device within the first frames. So a shader part is
// shared by pipelines whose two descriptor set layouts match, and all parts use one push constant
// range for every stage.
uint32_t CreateLibraryPipeline(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                               const GraphicsPipelineState&           state,
                               const PipelineCache::GraphicsPrograms& programs,
                               PipelineLibraryCache& libraries, vk::PipelineCache driver_cache) {
	using Part                       = vk::GraphicsPipelineLibraryFlagBitsEXT;
	const uint32_t pre_raster_stages = state.stage_count - state.fragment_stage_count;
	const vk::PipelineRenderingCreateInfo shader_rendering {};
	vk::PipelineDynamicStateCreateInfo    shared_dynamic_state {};
	shared_dynamic_state.dynamicStateCount = state.shared_dynamic_state_count;
	shared_dynamic_state.pDynamicStates    = state.dynamic_states.data();

	CreateGraphicsLayouts(graphics, pipeline, state, LibraryPushConstantStages);
	pipeline.push_constant_stages = LibraryPushConstantStages;
	std::string layout_key;
	AppendBindingsKey(layout_key, state.vertex_bindings, state.vertex_uses_push);
	AppendBindingsKey(layout_key, state.pixel_bindings, state.pixel_uses_push);

	std::string vertex_input_key(1, 'V');
	AppendKey(vertex_input_key, state.vertex_input.vertexBindingDescriptionCount);
	for (uint32_t i = 0; i < state.vertex_input.vertexBindingDescriptionCount; i++) {
		AppendKey(vertex_input_key, state.input_desc[i]);
	}
	AppendKey(vertex_input_key, state.vertex_input.vertexAttributeDescriptionCount);
	for (uint32_t i = 0; i < state.vertex_input.vertexAttributeDescriptionCount; i++) {
		AppendKey(vertex_input_key, state.input_attr[i]);
	}
	AppendKey(vertex_input_key, state.input_assembly.topology);
	AppendKey(vertex_input_key, state.input_assembly.primitiveRestartEnable);

	std::string pre_raster_key(1, 'P');
	AppendKey(pre_raster_key, pre_raster_stages);
	for (uint32_t i = 0; i < pre_raster_stages; i++) {
		AppendKey(pre_raster_key, programs.vertex[i].id);
		AppendKey(pre_raster_key, static_cast<uint32_t>(state.stages[i].stage));
	}
	AppendKey(pre_raster_key, state.uses_tessellation);
	AppendKey(pre_raster_key, state.tessellation.patchControlPoints);
	AppendKey(pre_raster_key, state.depth_clip_control.negativeOneToOne);
	AppendKey(pre_raster_key, state.depth_clip.depthClipEnable);
	AppendKey(pre_raster_key, state.provoking_vertex.provokingVertexMode);
	AppendKey(pre_raster_key, static_cast<uint32_t>(state.rasterizer.cullMode));
	AppendKey(pre_raster_key, state.rasterizer.frontFace);
	AppendKey(pre_raster_key, state.rasterizer.polygonMode);
	pre_raster_key += layout_key;

	// Without a depth attachment the bounds test has nothing to test.
	vk::PipelineDepthStencilStateCreateInfo depth_stencil {};
	if (state.with_depth && state.depth_stencil.depthBoundsTestEnable == VK_TRUE) {
		depth_stencil = state.depth_stencil;
	}
	std::string fragment_key(1, 'F');
	AppendKey(fragment_key, state.fragment_stage_count != 0 ? programs.pixel.id : uint64_t {0});
	AppendKey(fragment_key, state.multisampling.rasterizationSamples);
	AppendKey(fragment_key, state.multisampling.sampleShadingEnable);
	AppendKey(fragment_key, depth_stencil.depthBoundsTestEnable);
	AppendKey(fragment_key, depth_stencil.minDepthBounds);
	AppendKey(fragment_key, depth_stencil.maxDepthBounds);
	fragment_key += layout_key;

	std::string output_key(1, 'O');
	AppendKey(output_key, state.rendering.colorAttachmentCount);
	for (uint32_t i = 0; i < state.rendering.colorAttachmentCount; i++) {
		AppendKey(output_key, state.rendering.pColorAttachmentFormats[i]);
		AppendKey(output_key, state.blend[i]);
	}
	AppendKey(output_key, state.rendering.depthAttachmentFormat);
	AppendKey(output_key, state.rendering.stencilAttachmentFormat);
	AppendKey(output_key, state.multisampling.rasterizationSamples);
	AppendKey(output_key, state.multisampling.sampleShadingEnable);
	AppendKey(output_key, state.dynamic_state.dynamicStateCount);

	uint32_t    built           = 0;
	auto        vertex_input    = libraries.Find(vertex_input_key);
	auto        pre_raster      = libraries.Find(pre_raster_key);
	auto        fragment        = libraries.Find(fragment_key);
	auto        fragment_output = libraries.Find(output_key);
	const auto& layout          = pipeline.pipeline_layout;

	// The two shader parts hold nearly all of the compile time; when both are new, the fragment
	// part compiles on another thread while this one compiles the pre-rasterization part.
	std::future<vk::Pipeline>      fragment_compile;
	vk::GraphicsPipelineCreateInfo fragment_info {};
	if (fragment == nullptr) {
		fragment_info.stageCount = state.fragment_stage_count;
		fragment_info.pStages    = state.fragment_stage_count != 0
		                               ? state.stages.data() + pre_raster_stages
		                               : nullptr;
		fragment_info.pMultisampleState  = &state.multisampling;
		fragment_info.pDepthStencilState = &depth_stencil;
		fragment_info.pDynamicState      = &shared_dynamic_state;
		fragment_info.layout             = layout;
		const auto compile = [&] {
			return CreateLibraryPart(graphics, Part::eFragmentShader, fragment_info,
			                         shader_rendering, driver_cache);
		};
		if (pre_raster == nullptr) {
			fragment_compile = std::async(std::launch::async, compile);
		} else {
			fragment = libraries.Insert(std::move(fragment_key), compile());
			built |= 1u << 2u;
		}
	}
	if (pre_raster == nullptr) {
		vk::GraphicsPipelineCreateInfo info {};
		info.stageCount          = pre_raster_stages;
		info.pStages             = state.stages.data();
		info.pTessellationState  = state.uses_tessellation ? &state.tessellation : nullptr;
		info.pViewportState      = &state.viewport;
		info.pRasterizationState = &state.rasterizer;
		info.pDynamicState       = &shared_dynamic_state;
		info.layout              = layout;
		pre_raster = libraries.Insert(std::move(pre_raster_key),
		                              CreateLibraryPart(graphics, Part::ePreRasterizationShaders,
		                                                info, shader_rendering, driver_cache));
		built |= 1u << 1u;
	}
	if (fragment_compile.valid()) {
		fragment = libraries.Insert(std::move(fragment_key), fragment_compile.get());
		built |= 1u << 2u;
	}
	if (vertex_input == nullptr) {
		vk::GraphicsPipelineCreateInfo info {};
		info.pVertexInputState   = &state.vertex_input;
		info.pInputAssemblyState = &state.input_assembly;
		info.pDynamicState       = &shared_dynamic_state;
		vertex_input = libraries.Insert(std::move(vertex_input_key),
		                                CreateLibraryPart(graphics, Part::eVertexInputInterface,
		                                                  info, shader_rendering, driver_cache));
		built |= 1u << 0u;
	}
	if (fragment_output == nullptr) {
		vk::GraphicsPipelineCreateInfo info {};
		info.pMultisampleState = &state.multisampling;
		info.pColorBlendState  = &state.color_blending;
		info.pDynamicState     = &state.dynamic_state;
		fragment_output = libraries.Insert(std::move(output_key),
		                                   CreateLibraryPart(graphics, Part::eFragmentOutputInterface,
		                                                     info, state.rendering, driver_cache));
		built |= 1u << 3u;
	}

	const std::array parts {vertex_input, pre_raster, fragment, fragment_output};
	vk::PipelineLibraryCreateInfoKHR link_libraries {};
	link_libraries.libraryCount = static_cast<uint32_t>(parts.size());
	link_libraries.pLibraries   = parts.data();
	vk::GraphicsPipelineCreateInfo link {};
	link.pNext             = &link_libraries;
	link.layout            = layout;
	link.basePipelineIndex = -1;
	EXIT_IF(pipeline.pipeline != nullptr);
	EXIT_NOT_IMPLEMENTED(graphics.device.createGraphicsPipelines(nullptr, 1, &link, nullptr,
	                                                             &pipeline.pipeline) !=
	                     vk::Result::eSuccess);
	libraries.QueueOptimizedLink(&pipeline, parts, layout);
	pipeline.optimize_pending = true;
	return built;
}

} // namespace

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
int CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                           const PipelineRenderingState&          rendering,
                           const PipelineVertexInputState&        vertex_input,
                           std::span<const ShaderVertexInputInfo> vertex_info,
                           const ShaderPixelInputInfo*            ps_input_info,
                           const PipelineCache::GraphicsPrograms& programs,
                           const PipelineStaticParameters&        static_params,
                           PipelineLibraryCache* libraries, vk::PipelineCache driver_cache) {
	GraphicsPipelineState state(graphics.device);
	BuildGraphicsPipelineState(state, graphics, rendering, vertex_input, vertex_info, ps_input_info,
	                           programs, static_params);
	if (graphics_debug_dump_enabled()) {
		LOGF("PipelineTrace: vkCreateGraphicsPipelines begin VS=%" PRIu64 " PS=%" PRIu64
		     " topology=%" PRIu32 " color_mask=0x%08" PRIx32
		     " depth=%s blend=%s dyn_states=%" PRIu32 "\n",
		     programs.vertex[0].id, ps_input_info != nullptr ? programs.pixel.id : 0,
		     static_cast<uint32_t>(static_params.topology), static_params.color_mask[0],
		     (state.with_depth ? "true" : "false"),
		     (static_params.blend_enable[0] ? "true" : "false"),
		     state.dynamic_state.dynamicStateCount);
	}
	// Mesh and RectList pipelines stay monolithic: mesh shading has no library parts here, and
	// RectList tessellation shaders are generated per vertex/pixel shader pair. Static feedback
	// loop flags only appear where the dynamic feedback loop state is unsupported.
	if (libraries != nullptr && graphics.pipeline_library_enabled &&
	    graphics.pipeline_library_fast_linking && Config::PipelineLibrariesEnabled() &&
	    !state.mesh && !state.rect_list && !state.flags) {
		return static_cast<int>(
		    CreateLibraryPipeline(graphics, pipeline, state, programs, *libraries, driver_cache));
	}
	CreateMonolithicPipeline(graphics, pipeline, state, driver_cache);
	return -1;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const ShaderComputeInputInfo& input_info,
                            vk::ShaderModule compute_module, vk::PipelineCache driver_cache) {
	EXIT_IF(compute_module == nullptr);

	vk::PipelineShaderStageCreateInfo                     comp_shader_stage_info {};
	vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo comp_subgroup_size {};
	comp_shader_stage_info.stage  = vk::ShaderStageFlagBits::eCompute;
	comp_shader_stage_info.module = compute_module;
	comp_shader_stage_info.pName  = "main";
	EXIT_IF(!input_info.stage);
	const auto wave_size = input_info.stage.program->wave_size;
	if (graphics.compute_subgroup_size_control_enabled &&
	    wave_size >= graphics.min_subgroup_size && wave_size <= graphics.max_subgroup_size) {
		comp_subgroup_size.requiredSubgroupSize = wave_size;
		comp_shader_stage_info.pNext            = &comp_subgroup_size;
	}

	std::vector<vk::DescriptorSetLayoutBinding> descriptor_bindings;
	AddLayoutBindings(descriptor_bindings, *input_info.stage.program,
	                  vk::ShaderStageFlagBits::eCompute);
	CreateDescriptorLayout(graphics, pipeline, descriptor_bindings);
	const vk::PushConstantRange push_constants {vk::ShaderStageFlagBits::eCompute, 0,
	                                            ShaderRecompiler::IR::NativePushConstantSize};

	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = 1;
	pipeline_layout_info.pSetLayouts            = &pipeline.descriptor_set_layout;
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push_constants;

	EXIT_IF(pipeline.pipeline_layout != nullptr);

	LOGF("PipelineTrace: vkCreatePipelineLayout CS begin set_layouts=1 push_constants=%u\n",
	     1u);
	auto result = graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
	                                                  &pipeline.pipeline_layout);
	LOGF("PipelineTrace: vkCreatePipelineLayout CS done result=%s layout=%p\n",
	     vk::to_string(result).c_str(), static_cast<void*>(pipeline.pipeline_layout));
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	EXIT_NOT_IMPLEMENTED(pipeline.pipeline_layout == nullptr);

	vk::ComputePipelineCreateInfo info {};
	info.stage             = comp_shader_stage_info;
	info.layout            = pipeline.pipeline_layout;
	info.basePipelineIndex = -1;

	EXIT_IF(pipeline.pipeline != nullptr);

	LOGF("PipelineTrace: vkCreateComputePipelines begin layout=%p\n",
	     static_cast<void*>(pipeline.pipeline_layout));
	result = graphics.device.createComputePipelines(driver_cache, 1, &info, nullptr,
	                                                &pipeline.pipeline);
	LOGF("PipelineTrace: vkCreateComputePipelines done result=%s pipeline=%p\n",
	     vk::to_string(result).c_str(), static_cast<void*>(pipeline.pipeline));
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	EXIT_NOT_IMPLEMENTED(pipeline.pipeline == nullptr);
}

} // namespace Libs::Graphics
