#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_DRAWSPECULATION_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_DRAWSPECULATION_H_

#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "graphics/shader/shaderCompiler.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <xmmintrin.h>

namespace Libs::Graphics {

// One stage's resources, materialized ahead of its draw on the draw speculation thread (see
// DrawSpeculator), with every read that produced them. The GPU thread adopts them instead of
// materializing when the draw has the same program source, user data and shader base and every
// read still gives the same result (see ShaderRecompiler::IR::ReadsUnchanged).
struct SpeculatedStage {
	const void*                                  source = nullptr; // The program cache's entry.
	std::vector<uint32_t>                        user_data;
	uint64_t                                     shader_base = 0;
	ShaderRecompiler::IR::ResourceSnapshot       snapshot;
	ShaderRecompiler::IR::ResourceSpecialization specialization;
	ShaderRecompiler::IR::ReadLog                reads;
};

// What preparing a non-tessellated draw's stage inputs reads from registers (see
// PrepareGraphicsStages), copied whole where that is cheap. Filled with memset and memcpy so that
// equal registers give equal bytes: the speculation's registers replay the GPU thread's, so a
// difference means the replay went astray, or a field the preparation reads changed.
struct GraphicsStageRegisters {
	HW::VertexShaderInfo                           vertex;
	HW::PixelShaderInfo                            pixel;
	HW::ShaderRegisters                            shader;
	HW::BlendControl                               blend0;
	HW::ClipControl                                clip;
	HW::ModeControl                                mode;
	HW::GeControl                                  ge;
	std::array<float, 4>                           viewport0; // xscale, yscale, xoffset, yoffset
	std::array<Prospero::ColorComponentMapping, 8> export_mapping;
	uint32_t                                       shader_stages    = 0;
	uint32_t                                       prim_type        = 0;
	bool                                           rt0_blend_bypass = false;
	bool                                           pixel_active     = false;
};

// A draw's stage inputs, prepared ahead as GetGraphicsPrograms prepares them. The GPU thread
// takes them instead of preparing when its registers copy to the same bytes, no shader was
// registered since, and the vertex tables the preparation read are unchanged: preparation depends
// on nothing else.
struct PreparedGraphicsStages {
	bool                   valid = false;
	GraphicsStageRegisters registers;
	uint64_t               shader_map_version = 0;
	VertexTableReads       vertex_tables;
	ShaderVertexInputInfo  vertex_info;
	ShaderPixelInputInfo   pixel_info;
	ShaderParams           vertex_params;
	ShaderParams           pixel_params;
};

struct SpeculatedDraw {
	static constexpr uint32_t Vertex = 0; // The vertex or mesh stage of a non-tessellated draw.
	static constexpr uint32_t Pixel  = 1;
	// A stage without a source was not speculated.
	std::array<SpeculatedStage, 2> stages;
	PreparedGraphicsStages         prepared;
};

// The GPU thread's current draw, when it was speculated: set around the draw packet's handler.
inline thread_local SpeculatedDraw* t_speculated_draw = nullptr;

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_DRAWSPECULATION_H_ */
