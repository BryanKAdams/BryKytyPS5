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

// Brings a taken speculation into this core's caches before the draw reads it: the worker wrote
// it on another core, and the guest tables adoption reads again were last written by the game's
// threads, so each first touch would be a cross-core miss. The draw's setup runs meanwhile.
// Prefetches never fault, whatever the address.
inline void PrefetchSpeculatedDraw(const SpeculatedDraw& draw) {
	const auto lines = [](const void* data, size_t bytes) {
		const auto* bytes_start = static_cast<const char*>(data);
		for (size_t offset = 0; offset < bytes; offset += 64) {
			_mm_prefetch(bytes_start + offset, _MM_HINT_T0);
		}
	};
	const auto vector = [&](const auto& values) {
		lines(values.data(), values.size() * sizeof(values[0]));
	};
	for (const auto& stage: draw.stages) {
		if (stage.source == nullptr) {
			continue;
		}
		vector(stage.user_data);
		vector(stage.reads.entries);
		vector(stage.reads.words);
		for (const auto& entry: stage.reads.entries) {
			lines(reinterpret_cast<const void*>(entry.address), entry.count * sizeof(uint32_t));
		}
		vector(stage.snapshot.buffers);
		vector(stage.snapshot.images);
		vector(stage.snapshot.samplers);
		vector(stage.snapshot.flattened_srt);
		vector(stage.snapshot.user_data);
		vector(stage.specialization.buffers);
		vector(stage.specialization.images);
	}
	const auto& prepared = draw.prepared;
	if (prepared.valid) {
		lines(&prepared.registers, sizeof(prepared.registers));
		const auto& tables = prepared.vertex_tables;
		for (uint32_t i = 0; i < tables.count && i < tables.reads.size(); i++) {
			const auto& read = tables.reads[i];
			lines(reinterpret_cast<const void*>(read.address), read.dwords * sizeof(uint32_t));
			lines(tables.words.data() + read.first, read.dwords * sizeof(uint32_t));
		}
		const auto& vertex = prepared.vertex_info;
		lines(&vertex, offsetof(ShaderVertexInputInfo, resources_dst));
		lines(&vertex.stage, sizeof(vertex) - offsetof(ShaderVertexInputInfo, stage));
		lines(&prepared.pixel_info, sizeof(prepared.pixel_info));
		lines(&prepared.vertex_params, sizeof(prepared.vertex_params));
		lines(&prepared.pixel_params, sizeof(prepared.pixel_params));
	}
}

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_DRAWSPECULATION_H_ */
