#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <array>
#include <span>

namespace Libs::Graphics::ShaderRecompiler::IR {

class Value;

using SrtMemoryReader = bool (*)(void* userdata, uint64_t address, std::span<uint32_t> values);

struct SrtRuntime {
	std::span<const uint32_t> user_data;
	uint64_t                  shader_base                = 0;
	SrtMemoryReader           read_memory                = nullptr;
	void*                     userdata                   = nullptr;
	SrtMemoryReader           read_specialization_memory = nullptr;
	// read_specialization_memory succeeds for an aligned, nonzero 64-byte block only when every
	// dword in it would succeed with the same value, so one refresh may read whole blocks.
	bool                      specialization_block_reads = false;
	// Userdata for read_specialization_memory; null means userdata.
	void*                     specialization_userdata    = nullptr;
	// Optional, with read_memory: fills an aligned 64-byte block with what read_memory would
	// return dword by dword, or fails having read nothing. One refresh then serves its direct
	// reads from blocks and reads a block's dwords one by one after a failure.
	SrtMemoryReader           read_memory_block          = nullptr;
};

// One refresh's direct-read blocks (see SrtRuntime::read_memory_block), direct-mapped.
struct SrtDirectBlocks {
	static constexpr uint64_t BlockBytes = 64;
	static constexpr uint32_t Slots      = 16;

	struct Block {
		uint64_t                                         address = 0; // Zero is never a block.
		std::array<uint32_t, BlockBytes / sizeof(uint32_t)> words;
		bool                                             exact = false;
	};
	std::array<Block, Slots> blocks;
};

[[nodiscard]] inline void* SpecializationUserdata(const SrtRuntime& runtime) {
	return runtime.specialization_userdata != nullptr ? runtime.specialization_userdata
	                                                  : runtime.userdata;
}

enum class RuntimeValueType { Any, Integer };

// Collects reachable ReadConst values. Immediate offsets receive compact flat-buffer slots;
// dynamic offsets remain explicit and are never assigned a fake slot.
void BuildSrtPlan(Program& program);
bool ValidateRuntimeValue(const ResourcePlan& program, Value value,
                          RuntimeValueType type = RuntimeValueType::Any);
// Uses the strict reader for values that affect shader specialization.
SrtRuntime CleanRuntime(SrtRuntime runtime);

// One memoized evaluation session shared by the entire shader resource refresh.
class SrtWalker {
public:
	SrtWalker(const ResourcePlan& program, const SrtRuntime& runtime,
	          std::span<const uint8_t> clean_flat_slots = {}, SrtWalker* clean_evaluator = nullptr,
	          Value active_mask = {});
	~SrtWalker();
	SrtWalker(const SrtWalker&)            = delete;
	SrtWalker& operator=(const SrtWalker&) = delete;

	bool Evaluate(Value value, uint32_t& result);
	bool EvaluateDescriptor(uint32_t source, DescriptorValue& result);
	// An empty span means that all sources are active.
	std::span<const uint8_t> FindActiveSources();
	bool RefreshFlatBuffer(std::vector<uint32_t>& flat);

	static ResourcePlan::EvaluationContext& AcquireContext(const ResourcePlan& program);

private:
	static float Float32(uint64_t bits);
	bool EvaluateWide(Value value, uint64_t& result);
	bool Arg(const Inst& inst, size_t index, uint64_t& result);
	bool EvaluatePhi(const Inst& inst, uint64_t& result);
	bool EvaluateExtract(const Inst& inst, uint64_t& result);
	bool EvaluateRawRead(const Inst& inst, uint64_t& result);
	bool EvaluateInst(const Inst& inst, uint64_t& result);

	const ResourcePlan&              m_program;
	SrtRuntime                      m_runtime;
	std::span<const uint8_t>         m_clean_flat_slots;
	SrtWalker*                      m_clean_evaluator = nullptr;
	Value                           m_active_mask;
	ResourcePlan::EvaluationContext& m_context;
};

// Resolves a plan's descriptor, SRT, condition and fill values into index-based nodes. The plan
// must not be modified afterwards.
const CompiledResourcePlan& CompileResourcePlan(const ResourcePlan& program);

// SrtWalker over a CompiledResourcePlan. It keeps SrtWalker's evaluation contexts, clean
// predicates, EXEC-mask selects and failure semantics, so both produce identical results.
class SrtEvaluator {
public:
	SrtEvaluator(const ResourcePlan& program, const CompiledResourcePlan& compiled,
	             const SrtRuntime& runtime, bool clean_flat_slots = false,
	             SrtEvaluator* clean_evaluator = nullptr,
	             uint32_t      active_mask     = ResourceNode::NoNode);
	~SrtEvaluator();
	SrtEvaluator(const SrtEvaluator&)            = delete;
	SrtEvaluator& operator=(const SrtEvaluator&) = delete;

	bool Evaluate(uint32_t node, uint32_t& result);
	bool EvaluateDescriptor(uint32_t source, DescriptorValue& result);
	// An empty span means that all sources are active.
	std::span<const uint8_t> FindActiveSources();
	bool RefreshFlatBuffer(std::vector<uint32_t>& flat);
	// Serves this walker's own direct reads from blocks; needs SrtRuntime::read_memory_block.
	void UseDirectBlocks(SrtDirectBlocks& blocks) { m_direct_blocks = &blocks; }

private:
	bool EvaluateWide(uint32_t node, uint64_t& result);
	bool EvaluateInst(const ResourceNode& node, uint64_t& result);
	bool EvaluateRawRead(const ResourceNode& node, uint64_t& result);
	bool ReadDirectWord(uint64_t address, uint32_t& word);

	const ResourcePlan&              m_program;
	const CompiledResourcePlan&      m_compiled;
	const ResourceNode*              m_nodes;
	SrtRuntime                       m_runtime;
	bool                             m_clean_flat_slots = false;
	SrtEvaluator*                    m_clean_evaluator  = nullptr;
	uint32_t                         m_active_mask      = ResourceNode::NoNode;
	ResourcePlan::EvaluationContext& m_context;
	ResourcePlan::EvaluationContext::Entry* m_memo;
	uint64_t                         m_generation;
	// Set by a successful RefreshFlatBuffer: this walker's slot values.
	const std::vector<uint32_t>*     m_flat = nullptr;
	SrtDirectBlocks*                 m_direct_blocks = nullptr;
};

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_ */
