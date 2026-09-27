#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_

#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

namespace Libs::Graphics::ShaderRecompiler::IR {

// Canonical module-affecting resource state. Runtime addresses and descriptor payloads remain in
// ResourceSnapshot and therefore do not create shader permutations.
struct ResourceSpecialization {
	struct Buffer {
		uint32_t               packed_stride                   = 0;
		Prospero::BufferFormat descriptor_format               = Prospero::BufferFormat::kInvalid;
		uint32_t               descriptor_swizzle              = DstSel(4, 5, 6, 7);
		bool                   zero_stride_oob                 = false;
		bool                   operator==(const Buffer&) const = default;
	};

	struct Image {
		Prospero::TextureNumericClass numeric_class = Prospero::TextureNumericClass::Unsupported;
		Decoder::ImageDimension       dimension     = Decoder::ImageDimension::Unknown;
		uint32_t                      mip_count     = 1;
		Prospero::BufferFormat        conversion_format          = Prospero::BufferFormat::kInvalid;
		uint32_t                      shader_swizzle             = ShaderImageIdentitySwizzle;
		uint32_t                      indirect_root              = ImageResource::NoIndirectImage;
		uint32_t                      indirect_mapping_offset    = 0;
		uint32_t                      indirect_search_iterations = 0;
		bool                          cube                       = false;
		bool                          fmask                      = false;
		bool                          operator==(const Image&) const = default;
	};

	std::vector<Buffer> buffers;
	std::vector<Image>  images;

	bool operator==(const ResourceSpecialization&) const = default;
};

// Extracts the descriptor/SRT value graph before resource specialization. The returned plan owns
// its values and is independent of the translated shader CFG.
ResourcePlan ExtractResourcePlan(const Program& program);

// Inputs of the previous refresh of one plan into one snapshot and specialization. Pass the same
// memo only with the same plan, snapshot and specialization objects, and do not modify the
// snapshot's descriptors or the specialization between refreshes.
struct MaterializationMemo {
	std::vector<uint32_t> key;
	bool                  valid  = false;
	// The last refresh kept the previous descriptors and specialization.
	bool                  reused = false;
};

// Refreshes cached resources and specialization in place. A failed refresh must not be used.
// With a memo, a memoizable plan whose descriptor inputs (active sources, user data, shader base
// and flat SRT slots) are unchanged skips descriptor evaluation and specialization.
bool MaterializeResources(const ResourcePlan& program, const SrtRuntime& runtime,
                          ResourceSnapshot& snapshot, ResourceSpecialization& specialization,
                          MaterializationMemo* memo = nullptr);

// The guest reads one refresh made through its runtime's readers, in order, with their results.
struct ReadLog {
	enum class Reader : uint8_t { Ordinary, OrdinaryBlock, Specialization };
	struct Entry {
		uint64_t address = 0;
		uint32_t first   = 0; // Into words.
		uint32_t count   = 0; // Dwords.
		Reader   reader  = Reader::Ordinary;
		bool     ok      = false;
	};

	std::vector<Entry>    entries;
	std::vector<uint32_t> words;

	void Clear() {
		entries.clear();
		words.clear();
	}
};

// MaterializeResources without a memo, logging every read it makes through runtime's readers.
// Plans that capture their reads (masked indirect images beside written buffers) are refused.
bool MaterializeResourcesLogged(const ResourcePlan& program, const SrtRuntime& runtime,
                                ResourceSnapshot& snapshot, ResourceSpecialization& specialization,
                                ReadLog& log);

// Whether every logged read, made again through runtime's readers, fails as it failed or reads
// the same dwords. A refresh is a function of its plan, user data, shader base and read results,
// so a refresh with this runtime (the same plan, user data and shader base) would then produce
// the logged refresh's snapshot and specialization.
bool ReadsUnchanged(const ReadLog& log, const SrtRuntime& runtime);

// The SrtWalker implementation that MaterializeResources must match.
bool MaterializeResourcesReference(const ResourcePlan& program, const SrtRuntime& runtime,
                                   ResourceSnapshot&       snapshot,
                                   ResourceSpecialization& specialization);

// Debugging aid: compare every refresh with MaterializeResourcesReference and abort on a
// difference. GPU thread only.
void SetResourceMaterializationVerification(bool enabled);

// Applies an already-derived specialization to native IR before layout and emission.
void ApplyResourceSpecialization(Program& program, const ResourceSpecialization& specialization);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_ */
