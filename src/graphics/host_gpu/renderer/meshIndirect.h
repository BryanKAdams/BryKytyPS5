#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MESHINDIRECT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MESHINDIRECT_H_

#include "common/common.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <cstdint>

namespace Libs::Graphics {

struct GraphicContext;

// Turns a GPU-written guest DRAW_INDEX_INDIRECT argument block into a mesh draw on the GPU. A
// one-thread compute pass writes the draw's parameter record and a
// VkDrawMeshTasksIndirectCommandEXT, so the CPU never reads the arguments.
class MeshIndirectArgs {
public:
	// Output layout: the 7-dword parameter record at offset 0, the command at CommandOffset.
	static constexpr uint64_t RecordOffset  = 0;
	static constexpr uint64_t CommandOffset = 32;
	static constexpr uint64_t OutputSize    = 48;
	static constexpr uint64_t ArgumentsSize = 20;

	struct Params {
		uint32_t index_bytes          = 0;
		uint64_t index_address        = 0;
		uint32_t index_limit          = 0; // INDEX_BUFFER_SIZE in elements; 0 when unknown.
		uint32_t primitive_size       = 0;
		uint32_t primitive_step       = 0;
		uint32_t primitives_per_group = 0;
		uint32_t max_groups           = 0;
		uint32_t max_instances        = 0;
		uint32_t max_total            = 0;
		// Write a VkDrawIndexedIndirectCommand with the count clamped, for a draw that is not
		// mesh-emulated; the primitive and host-limit fields are unused.
		bool     plain                = false;
	};

	explicit MeshIndirectArgs(GraphicContext& graphics);
	~MeshIndirectArgs();
	KYTY_CLASS_NO_COPY(MeshIndirectArgs);

	// Records the conversion and the barriers that order it after earlier GPU writes of the
	// arguments and before the indirect mesh draw. Must be recorded outside a rendering instance.
	void Record(vk::CommandBuffer command, const Buffer& arguments, uint64_t arguments_offset,
	            const Buffer& output, uint64_t output_offset, const Params& params);

private:
	GraphicContext&         m_graphics;
	vk::DescriptorSetLayout m_descriptor_layout = nullptr;
	vk::PipelineLayout      m_pipeline_layout   = nullptr;
	vk::Pipeline            m_pipeline          = nullptr;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MESHINDIRECT_H_
