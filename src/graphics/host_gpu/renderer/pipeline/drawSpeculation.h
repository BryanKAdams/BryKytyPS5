#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_DRAWSPECULATION_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_DRAWSPECULATION_H_

#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include <array>
#include <cstdint>
#include <vector>

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

struct SpeculatedDraw {
	static constexpr uint32_t Vertex = 0; // The vertex or mesh stage of a non-tessellated draw.
	static constexpr uint32_t Pixel  = 1;
	// A stage without a source was not speculated.
	std::array<SpeculatedStage, 2> stages;
};

// The GPU thread's current draw, when it was speculated: set around the draw packet's handler.
inline thread_local SpeculatedDraw* t_speculated_draw = nullptr;

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_DRAWSPECULATION_H_ */
