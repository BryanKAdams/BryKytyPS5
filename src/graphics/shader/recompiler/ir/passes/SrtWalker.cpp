#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include "common/assert.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <fmt/format.h>
#include <map>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace Libs::Graphics::ShaderRecompiler::IR {

SrtRuntime CleanRuntime(SrtRuntime runtime) {
	if (runtime.read_specialization_memory != nullptr) {
		runtime.read_memory = runtime.read_specialization_memory;
		runtime.userdata    = SpecializationUserdata(runtime);
	} else {
		runtime.read_memory = +[](void*, uint64_t, std::span<uint32_t>) { return false; };
	}
	return runtime;
}

namespace {

constexpr uint64_t AddressMask = 0x0000ffffffffffffull;

const char* StageName(ShaderType stage) {
	switch (stage) {
		case ShaderType::Vertex: return "vertex";
		case ShaderType::Pixel: return "pixel";
		case ShaderType::Fetch: return "fetch";
		case ShaderType::Compute: return "compute";
		default: return "unknown";
	}
}

std::string Diagnostic(const ResourcePlan& program, uint32_t pc, const std::string& message) {
	return fmt::format("shader SRT: hash=0x{:016x} stage={} pc=0x{:08x} {}", program.shader_hash,
	                   StageName(program.stage), pc, message);
}

bool AddSignedAddress(uint64_t base, int64_t offset, uint64_t& result) {
	if (base > AddressMask) {
		return false;
	}
	if (offset < 0) {
		const auto magnitude = uint64_t {0} - static_cast<uint64_t>(offset);
		if (magnitude > base) {
			return false;
		}
		result = base - magnitude;
		return true;
	}
	const auto magnitude = static_cast<uint64_t>(offset);
	if (magnitude > AddressMask - base) {
		return false;
	}
	result = base + magnitude;
	return true;
}

bool IsRawRead(const ResourcePlan& values, const Inst& inst) {
	const auto op = inst.GetOpcode();
	if (op != ValueOpcode::LoadAddressU32 && op != ValueOpcode::ReadConstBuffer) {
		return false;
	}
	const auto index = inst.Flags<MemoryFlags>().index;
	if (index >= values.memory_info.size()) {
		return false;
	}
	const auto kind = values.memory_info[index].kind;
	return (op == ValueOpcode::LoadAddressU32 && kind == ResourceKind::ScalarAddress) ||
	       (op == ValueOpcode::ReadConstBuffer && kind == ResourceKind::ScalarBuffer);
}

bool IsDescriptorHandle(ValueOpcode opcode) {
	switch (opcode) {
		case ValueOpcode::GetBufferResource:
		case ValueOpcode::GetAddressResource:
		case ValueOpcode::GetImageResource:
		case ValueOpcode::GetSamplerResource: return true;
		default: return false;
	}
}

bool IsRuntimeSelect(ValueOpcode op) {
	return op == ValueOpcode::SelectU1 || op == ValueOpcode::SelectU32 ||
	       op == ValueOpcode::SelectF32;
}

bool IsRuntimeUniformOp(ValueOpcode op) {
	switch (op) {
		case ValueOpcode::BitCastU32F32:
		case ValueOpcode::BitCastF32U32:
		case ValueOpcode::ConvertU32F32:
		case ValueOpcode::ConvertF32U32:
		case ValueOpcode::CompositeConstructU64:
		case ValueOpcode::CompositeExtractU64:
		case ValueOpcode::CompositeConstructU32x2:
		case ValueOpcode::CompositeExtractU32x2:
		case ValueOpcode::BitFieldInsert:
		case ValueOpcode::BitFieldUExtract:
		case ValueOpcode::BitFieldSExtract:
		case ValueOpcode::IAdd32:
		case ValueOpcode::IAdd64:
		case ValueOpcode::IAddCarry32:
		case ValueOpcode::ISub32:
		case ValueOpcode::ISub64:
		case ValueOpcode::IMul32:
		case ValueOpcode::IMul64:
		case ValueOpcode::UMin32:
		case ValueOpcode::ShiftLeftLogical32:
		case ValueOpcode::ShiftLeftLogical64:
		case ValueOpcode::ShiftRightLogical32:
		case ValueOpcode::ShiftRightLogical64:
		case ValueOpcode::ShiftRightArithmetic32:
		case ValueOpcode::ShiftRightArithmetic64:
		case ValueOpcode::BitwiseAnd32:
		case ValueOpcode::BitwiseAnd64:
		case ValueOpcode::BitwiseOr32:
		case ValueOpcode::BitwiseXor32:
		case ValueOpcode::BitwiseNot32:
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectF32:
		case ValueOpcode::ULessThan32:
		case ValueOpcode::IEqual32:
		case ValueOpcode::UGreaterThan32:
		case ValueOpcode::SGreaterThanEqual32:
		case ValueOpcode::INotEqual32:
		case ValueOpcode::LogicalOr:
		case ValueOpcode::LogicalAnd:
		case ValueOpcode::LogicalXor:
		case ValueOpcode::LogicalNot:
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPOrdGreaterThanEqual32:
		case ValueOpcode::FPIsNan32:
		case ValueOpcode::FPMul32:
		case ValueOpcode::FPTrunc32: return true;
		default: return false;
	}
}

class RuntimeValidator {
public:
	explicit RuntimeValidator(const ResourcePlan& program, RuntimeValueType type)
	    : m_program(program), m_type(type) {}

	bool Run(Value value) { return Validate(value); }

private:
	bool ValidateArguments(const Inst& inst, bool require_uniform) {
		for (size_t index = 0; index < inst.NumArgs(); index++) {
			if (!Validate(inst.Arg(index), require_uniform)) return false;
		}
		return true;
	}

	bool Validate(Value value, bool require_uniform = true) {
		value = value.Resolve();
		// Host floating-point evaluation does not model shader rounding/denormal modes.
		if (m_type == RuntimeValueType::Integer &&
		    TypesOverlap(value.GetType(), Type::F16 | Type::F32 | Type::F32x2)) {
			return false;
		}
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) {
			if (!require_uniform) return true;
			switch (value.GetType()) {
				case Type::U1:
				case Type::U8:
				case Type::U16:
				case Type::U32:
				case Type::U64:
				case Type::F32: return true;
				default: return false;
			}
		}
		// Integer-only dependency checks do not depend on the active EXEC mask.
		if (!require_uniform && m_validated_dependencies.contains(inst)) return true;
		if (!m_visiting.insert(inst).second) {
			return !require_uniform;
		}
		const auto finish = [&](bool valid) {
			m_visiting.erase(inst);
			if (valid && !require_uniform) m_validated_dependencies.insert(inst);
			return valid;
		};
		const auto op = inst->GetOpcode();
		if (op == ValueOpcode::ReadConst) {
			const auto slot = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (inst->NumArgs() != 2 || inst->Arg(0).Resolve().TryInstruction() == nullptr ||
			    inst->Arg(0).Resolve().TryInstruction()->GetOpcode() !=
			        ValueOpcode::GetSrtResource ||
			    !slot.IsImmediate() || slot.GetType() != Type::U32 ||
			    slot.U32() >= m_program.srt_reads.size()) {
				return finish(false);
			}
			if (m_type == RuntimeValueType::Integer) {
				const auto active_mask = m_active_mask;
				m_active_mask          = {};
				const bool valid       = Validate(m_program.srt_reads[slot.U32()].value);
				m_active_mask          = active_mask;
				if (!valid) return finish(false);
			}
		}
		if (!require_uniform) return finish(ValidateArguments(*inst, false));
		if (!m_active_mask.IsEmpty() && IsRuntimeSelect(op) && inst->NumArgs() == 3 &&
		    inst->Arg(0).Resolve() == m_active_mask) {
			// Empty EXEC reads lane zero, so ignored operands still require integer types.
			if (m_type == RuntimeValueType::Integer && !Validate(inst->Arg(2), false)) {
				return finish(false);
			}
			return finish(Validate(inst->Arg(1)));
		}
		if (op == ValueOpcode::UndefU1 || op == ValueOpcode::UndefU8 ||
		    op == ValueOpcode::UndefU16 || op == ValueOpcode::UndefU32 ||
		    op == ValueOpcode::UndefU64 || op == ValueOpcode::Void) {
			return finish(false);
		}
		if (op == ValueOpcode::GetUserData) {
			if (inst->NumArgs() != 1 || inst->Arg(0).GetType() != Type::ScalarReg) {
				return finish(false);
			}
			const auto reg = RegIndex(inst->Arg(0).ScalarRegister());
			if (reg < m_program.user_data_base ||
			    reg - m_program.user_data_base >= m_program.user_data_count) {
				return finish(false);
			}
			return finish(true);
		}
		if (op == ValueOpcode::GetShaderBase) {
			if (inst->NumArgs() != 0) {
				return finish(false);
			}
			return finish(true);
		}
		if (op == ValueOpcode::Phi) {
			if (m_type == RuntimeValueType::Integer && !ValidateArguments(*inst, false)) {
				return finish(false);
			}
			const auto invariant = ResolveInvariantPhi(m_program, value);
			if (invariant.IsEmpty()) {
				return finish(false);
			}
			return finish(Validate(invariant));
		}
		if (op == ValueOpcode::ReadFirstLane) {
			if (inst->NumArgs() != 2 || inst->Arg(0).GetType() != Type::U32 ||
			    inst->Arg(1).GetType() != Type::U1) {
				return finish(false);
			}
			if (m_type == RuntimeValueType::Integer && !Validate(inst->Arg(1), false)) {
				return finish(false);
			}
			const auto active_mask = m_active_mask;
			m_active_mask          = inst->Arg(1).Resolve();
			const bool valid       = Validate(inst->Arg(0));
			m_active_mask          = active_mask;
			return finish(valid);
		}
		if (op == ValueOpcode::GetSrtResource) {
			if (inst->NumArgs() != 0) {
				return finish(false);
			}
			return finish(true);
		}
		if (op == ValueOpcode::LoadAddressU32 || op == ValueOpcode::ReadConstBuffer) {
			const auto  expected = op == ValueOpcode::LoadAddressU32
			                           ? ValueOpcode::GetAddressResource
			                           : ValueOpcode::GetBufferResource;
			const auto* handle = inst->NumArgs() != 0 ? inst->Arg(0).ResolveInstruction() : nullptr;
			if (!IsRawRead(m_program, *inst) || handle == nullptr ||
			    handle->GetOpcode() != expected) {
				return finish(false);
			}
		} else if (op == ValueOpcode::CompositeExtractU64) {
			const auto index = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (!index.IsImmediate() || index.GetType() != Type::U32 || index.U32() >= 2u) {
				return finish(false);
			}
		} else if (op == ValueOpcode::CompositeExtractU32x2) {
			const auto* source = inst->NumArgs() == 2 ? inst->Arg(0).ResolveInstruction() : nullptr;
			const auto  index  = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (source == nullptr || !index.IsImmediate() || index.GetType() != Type::U32 ||
			    index.U32() >= 2u ||
			    (source->GetOpcode() != ValueOpcode::CompositeConstructU32x2 &&
			     source->GetOpcode() != ValueOpcode::IAddCarry32)) {
				return finish(false);
			}
		}
		if (IsDescriptorHandle(op)) {
			size_t expected = 4u;
			if (op == ValueOpcode::GetImageResource) {
				expected = 8u;
			} else if (op == ValueOpcode::GetAddressResource) {
				expected = 2u;
			}
			if (inst->NumArgs() != expected) {
				return finish(false);
			}
		} else if (op != ValueOpcode::ReadConst && op != ValueOpcode::ReadConstBuffer &&
		           op != ValueOpcode::LoadAddressU32 && !IsRuntimeUniformOp(op)) {
			return finish(false);
		}
		return finish(ValidateArguments(*inst, true));
	}

	const ResourcePlan&             m_program;
	RuntimeValueType                m_type;
	Value                           m_active_mask;
	std::unordered_set<const Inst*> m_visiting;
	std::unordered_set<const Inst*> m_validated_dependencies;
};

class PlanBuilder {
public:
	explicit PlanBuilder(Program& program): m_program(program) {}

	void Run() {
		m_program.srt_reads.clear();
		m_program.dynamic_reads.clear();
		for (auto* block: m_program.blocks) {
			for (auto& inst: *block) {
				const auto op = inst.GetOpcode();
				if (op == ValueOpcode::LoadAddressU32 || op == ValueOpcode::ReadConstBuffer) {
					const auto flags = inst.Flags<MemoryFlags>();
					if (flags.index < m_program.memory_info.size()) {
						const auto kind       = m_program.memory_info[flags.index].kind;
						const bool crosswired = (op == ValueOpcode::LoadAddressU32 &&
						                         kind == ResourceKind::ScalarBuffer) ||
						                        (op == ValueOpcode::ReadConstBuffer &&
						                         kind == ResourceKind::ScalarAddress);
						if (crosswired) {
							Fail(flags.pc,
							     fmt::format("{} has incompatible scalar memory metadata",
							                 ValueOpcodeName(op)));
						}
					}
				}
				if (IsDescriptorHandle(inst.GetOpcode())) {
					for (size_t index = 0; index < inst.NumArgs(); index++) {
						Collect(inst.Arg(index), 0);
					}
				}
			}
		}
		for (auto* block: m_program.blocks) {
			for (auto& inst: *block) {
				if (inst.GetOpcode() == ValueOpcode::LoadAddressU32 && IsRawRead(m_program, inst) &&
				    inst.Arg(1).Resolve().IsImmediate() &&
				    ValidateRuntimeValue(m_program, Value(&inst))) {
					Collect(Value(&inst), inst.Flags<MemoryFlags>().pc);
				}
			}
		}
		PatchReads();
	}

private:
	struct Patch {
		Inst*    inst = nullptr;
		uint32_t slot = 0;
		bool     keep = false;
	};

	[[noreturn]] void Fail(uint32_t pc, const std::string& message) const {
		const auto diagnostic = Diagnostic(m_program, pc, message);
		EXIT("shader SRT planning failed: %s", diagnostic.c_str());
		std::abort();
	}

	void Collect(Value value, uint32_t use_pc) {
		value = value.Resolve();
		if (value.IsImmediate()) {
			return;
		}
		auto* inst = value.TryInstruction();
		if (inst == nullptr) {
			Fail(use_pc, "invalid typed planning value");
		}
		const auto cycle = std::ranges::find(m_visiting, inst);
		if (cycle != m_visiting.end()) {
			const auto contains_phi = std::any_of(cycle, m_visiting.end(), [](const Inst* value) {
				return value->GetOpcode() == ValueOpcode::Phi;
			});
			if (contains_phi) {
				return;
			}
			Fail(use_pc, fmt::format("cyclic typed planning value {} without a phi",
			                         ValueOpcodeName(inst->GetOpcode())));
		}
		if (std::ranges::find(m_visited, inst) != m_visited.end()) {
			return;
		}
		m_visiting.push_back(inst);
		for (size_t index = 0; index < inst->NumArgs(); index++) {
			Collect(inst->Arg(index), use_pc);
		}
		m_visiting.pop_back();
		m_visited.push_back(inst);
		if (!IsRawRead(m_program, *inst)) {
			return;
		}
		const auto offset = inst->Arg(1).Resolve();
		if (!offset.IsImmediate() || offset.GetType() != Type::U32) {
			if (std::ranges::find(m_program.dynamic_reads, value) ==
			    m_program.dynamic_reads.end()) {
				m_program.dynamic_reads.push_back(value);
			}
			return;
		}
		for (uint32_t slot = 0; slot < m_program.srt_reads.size(); slot++) {
			if (EquivalentValue(m_program, value, m_program.srt_reads[slot].value)) {
				m_patches.push_back({inst, slot, false});
				return;
			}
		}
		const auto slot = static_cast<uint32_t>(m_program.srt_reads.size());
		m_program.srt_reads.push_back({value, slot});
		m_patches.push_back({inst, slot, true});
	}

	void PatchReads() {
		for (const auto& patch: m_patches) {
			auto* block = patch.inst->Parent();
			auto& list  = block->Instructions();
			auto  where =
			    std::ranges::find_if(list, [&](const Inst& inst) { return &inst == patch.inst; });
			const auto resource =
			    Value(&*block->PrependNewInst(where, ValueOpcode::GetSrtResource));
			const auto flat = Value(&*block->PrependNewInst(where, ValueOpcode::ReadConst,
			                                                {resource, Value(patch.slot)}));
			const auto uses = patch.inst->Uses();
			for (const auto& use: uses) {
				use.user->SetArg(use.operand, flat);
			}
			for (auto& info: m_program.block_info) {
				if (info.condition.Resolve() == Value(patch.inst)) {
					info.condition = flat;
				}
				if (info.indirect_target.Resolve() == Value(patch.inst)) {
					info.indirect_target = flat;
				}
			}
			if (patch.keep) {
				const auto memory = patch.inst->Flags<MemoryFlags>().index;
				if (memory < m_program.memory_info.size()) {
					m_program.memory_info[memory].planning_only = true;
				}
				block->AppendNewInst(ValueOpcode::ReferenceU32, {Value(patch.inst)});
			}
		}
	}

	Program&           m_program;
	std::vector<Inst*> m_visiting;
	std::vector<Inst*> m_visited;
	std::vector<Patch> m_patches;
};

} // namespace

SrtWalker::SrtWalker(const ResourcePlan& program, const SrtRuntime& runtime,
                     std::span<const uint8_t> clean_flat_slots, SrtWalker* clean_evaluator,
                     Value active_mask)
    : m_program(program), m_runtime(runtime), m_clean_flat_slots(clean_flat_slots),
      m_clean_evaluator(clean_evaluator), m_active_mask(active_mask.Resolve()),
      m_context(AcquireContext(program)) {}

SrtWalker::~SrtWalker() { --m_program.evaluation_depth; }

bool SrtWalker::Evaluate(Value value, uint32_t& result) {
	uint64_t wide = 0;
	if (!EvaluateWide(value, wide)) {
		return false;
	}
	result = static_cast<uint32_t>(wide);
	return true;
}

ResourcePlan::EvaluationContext& SrtWalker::AcquireContext(const ResourcePlan& program) {
	if (program.evaluation_depth == program.evaluation_contexts.size()) {
		program.evaluation_contexts.emplace_back();
	}
	auto& context = program.evaluation_contexts[program.evaluation_depth++];
	context.generation += 2;
	return context;
}

float SrtWalker::Float32(uint64_t bits) {
	return std::bit_cast<float>(static_cast<uint32_t>(bits));
}

bool SrtWalker::EvaluateWide(Value value, uint64_t& result) {
	value = value.Resolve();
	if (value.IsImmediate()) {
		switch (value.GetType()) {
			case Type::U1: result = value.U1(); return true;
			case Type::U8: result = value.U8(); return true;
			case Type::U16: result = value.U16(); return true;
			case Type::U32: result = value.U32(); return true;
			case Type::U64: result = value.U64(); return true;
			case Type::F32: result = std::bit_cast<uint32_t>(value.F32Value()); return true;
			default: return false;
		}
	}
	auto* inst = value.TryInstruction();
	if (inst == nullptr) {
		return false;
	}
	if (!m_active_mask.IsEmpty() && IsRuntimeSelect(inst->GetOpcode()) &&
	    inst->NumArgs() == 3 && inst->Arg(0).Resolve() == m_active_mask) {
		return EvaluateWide(inst->Arg(1), result);
	}
	const auto index = inst->EvaluationIndex(m_program.evaluation_value_count);
	if (index >= m_context.values.size()) {
		m_context.values.resize(m_program.evaluation_value_count);
	}
	if (m_context.values[index].generation == m_context.generation) {
		result = m_context.values[index].value;
		return true;
	}
	// The low generation bit marks an instruction that is still being evaluated.
	if (m_context.values[index].generation == (m_context.generation | 1u)) {
		return false;
	}
	m_context.values[index].generation = m_context.generation | 1u;
	uint64_t out = 0;
	const bool evaluated = EvaluateInst(*inst, out);
	// Recursive evaluation may grow the dense memo vector.
	auto& memo = m_context.values[index];
	if (!evaluated) {
		memo.generation = 0;
		return false;
	}
	memo.value      = out;
	memo.generation = m_context.generation;
	result = out;
	return true;
}

bool SrtWalker::Arg(const Inst& inst, size_t index, uint64_t& result) {
	return EvaluateWide(inst.Arg(index), result);
}

bool SrtWalker::EvaluatePhi(const Inst& inst, uint64_t& result) {
	const auto value = ResolveInvariantPhi(m_program, Value(const_cast<Inst*>(&inst)));
	return !value.IsEmpty() && EvaluateWide(value, result);
}

bool SrtWalker::EvaluateExtract(const Inst& inst, uint64_t& result) {
	const auto index = inst.Arg(1).Resolve();
	if (!index.IsImmediate() || index.GetType() != Type::U32) {
		return false;
	}
	const auto component = index.U32();
	if (component >= 2u) {
		return false;
	}
	if (inst.GetOpcode() == ValueOpcode::CompositeExtractU64) {
		uint64_t packed = 0;
		if (!Arg(inst, 0, packed)) {
			return false;
		}
		result = static_cast<uint32_t>(packed >> (component * 32u));
		return true;
	}
	const auto* source = inst.Arg(0).ResolveInstruction();
	if (source == nullptr) {
		return false;
	}
	if (source->GetOpcode() == ValueOpcode::CompositeConstructU32x2) {
		return EvaluateWide(source->Arg(component), result);
	}
	if (source->GetOpcode() == ValueOpcode::IAddCarry32) {
		uint64_t lhs = 0;
		uint64_t rhs = 0;
		if (!Arg(*source, 0, lhs) || !Arg(*source, 1, rhs)) {
			return false;
		}
		const auto sum =
		    static_cast<uint64_t>(static_cast<uint32_t>(lhs)) + static_cast<uint32_t>(rhs);
		result =
		    component == 0u ? static_cast<uint32_t>(sum) : static_cast<uint32_t>(sum >> 32u);
		return true;
	}
	return false;
}

bool SrtWalker::EvaluateRawRead(const Inst& inst, uint64_t& result) {
	const auto flags = inst.Flags<MemoryFlags>();
	if (flags.index >= m_program.memory_info.size()) {
		return false;
	}
	const auto& mem    = m_program.memory_info[flags.index];
	const auto* handle = inst.Arg(0).ResolveInstruction();
	if (handle == nullptr) {
		return false;
	}
	uint64_t low    = 0;
	uint64_t high   = 0;
	uint64_t offset = 0;
	if (!Arg(*handle, 0, low) || !Arg(*handle, 1, high) || !Arg(inst, 1, offset)) {
		return false;
	}
	const auto base      = ((high << 32u) | static_cast<uint32_t>(low)) & AddressMask;
	const auto immediate = static_cast<int64_t>(static_cast<int32_t>(mem.offset));
	uint64_t   address   = 0;
	if (inst.GetOpcode() == ValueOpcode::ReadConstBuffer) {
		uint64_t records = 0;
		uint64_t word3   = 0;
		if (handle->NumArgs() != 4u || !Arg(*handle, 2, records) || !Arg(*handle, 3, word3)) {
			return false;
		}
		if (immediate < 0) {
			return false;
		}
		const auto byte_offset =
		    static_cast<uint64_t>(immediate) + static_cast<uint32_t>(offset);
		const auto aligned = byte_offset & ~uint64_t {3};
		const auto stride  = (static_cast<uint32_t>(high) >> 16u) & 0x3fffu;
		const auto size = stride == 0u
		                      ? static_cast<uint64_t>(static_cast<uint32_t>(records))
		                      : static_cast<uint64_t>(stride) * static_cast<uint32_t>(records);
		if (aligned > size || size - aligned < sizeof(uint32_t)) {
			return false;
		}
		address = ((base & ~uint64_t {3}) + byte_offset) & ~uint64_t {3};
	} else {
		const auto relative = (immediate & ~int64_t {3}) +
		                      static_cast<int64_t>(static_cast<uint32_t>(offset) & ~3u);
		if (!AddSignedAddress(base & ~uint64_t {3}, relative, address)) {
			return false;
		}
	}
	uint32_t word = 0;
	if (m_runtime.read_memory != nullptr) {
		if (!m_runtime.read_memory(m_runtime.userdata, address, {&word, 1})) {
			return false;
		}
	} else {
		std::memcpy(&word, reinterpret_cast<const void*>(address), sizeof(word));
	}
	result = word;
	return true;
}

bool SrtWalker::EvaluateInst(const Inst& inst, uint64_t& result) {
	uint64_t   a       = 0;
	uint64_t   b       = 0;
	uint64_t   c       = 0;
	const auto binary  = [&]() { return Arg(inst, 0, a) && Arg(inst, 1, b); };
	const auto ternary = [&]() {
		return Arg(inst, 0, a) && Arg(inst, 1, b) && Arg(inst, 2, c);
	};
	switch (inst.GetOpcode()) {
		case ValueOpcode::GetUserData: {
			const auto reg = RegIndex(inst.Arg(0).ScalarRegister());
			if (reg < m_program.user_data_base ||
			    reg - m_program.user_data_base >= m_runtime.user_data.size()) {
				return false;
			}
			result = m_runtime.user_data[reg - m_program.user_data_base];
			return true;
		}
		case ValueOpcode::GetShaderBase: result = m_runtime.shader_base; return true;
		case ValueOpcode::Phi: return EvaluatePhi(inst, result);
		case ValueOpcode::ReadFirstLane: {
			const auto clean_runtime = CleanRuntime(m_runtime);
			SrtWalker  clean_active(m_program, clean_runtime, {}, nullptr, inst.Arg(1));
			SrtWalker  active(m_program, m_runtime, m_clean_flat_slots, &clean_active,
			                  inst.Arg(1));
			return active.EvaluateWide(inst.Arg(0), result);
		}
		case ValueOpcode::BitCastU32F32:
		case ValueOpcode::BitCastF32U32: return Arg(inst, 0, result);
		case ValueOpcode::CompositeExtractU64:
		case ValueOpcode::CompositeExtractU32x2: return EvaluateExtract(inst, result);
		case ValueOpcode::CompositeConstructU64:
			if (!binary()) {
				return false;
			}
			result = static_cast<uint32_t>(a) |
			         (static_cast<uint64_t>(static_cast<uint32_t>(b)) << 32u);
			return true;
		case ValueOpcode::ReadConst: {
			const auto slot = inst.Arg(1).Resolve();
			if (!slot.IsImmediate() || slot.GetType() != Type::U32 ||
			    slot.U32() >= m_program.srt_reads.size()) {
				return false;
			}
			if (slot.U32() < m_clean_flat_slots.size() &&
			    m_clean_flat_slots[slot.U32()] != 0u && m_clean_evaluator != nullptr) {
				return m_clean_evaluator->EvaluateWide(m_program.srt_reads[slot.U32()].value,
				                                       result);
			}
			return EvaluateWide(m_program.srt_reads[slot.U32()].value, result);
		}
		case ValueOpcode::LoadAddressU32:
		case ValueOpcode::ReadConstBuffer:
			if (IsRawRead(m_program, inst)) {
				return EvaluateRawRead(inst, result);
			}
			break;
		case ValueOpcode::IAdd32:
			if (binary()) {
				result = static_cast<uint32_t>(a + b);
				return true;
			}
			return false;
		case ValueOpcode::IAdd64:
			if (binary()) {
				result = a + b;
				return true;
			}
			return false;
		case ValueOpcode::ISub32:
			if (binary()) {
				result = static_cast<uint32_t>(a - b);
				return true;
			}
			return false;
		case ValueOpcode::ISub64:
			if (binary()) {
				result = a - b;
				return true;
			}
			return false;
		case ValueOpcode::IMul32:
			if (binary()) {
				result = static_cast<uint32_t>(a * b);
				return true;
			}
			return false;
		case ValueOpcode::IMul64:
			if (binary()) {
				result = a * b;
				return true;
			}
			return false;
		case ValueOpcode::UMin32:
			if (binary()) {
				result = std::min(static_cast<uint32_t>(a), static_cast<uint32_t>(b));
				return true;
			}
			return false;
		case ValueOpcode::ConvertF32U32:
			if (Arg(inst, 0, a)) {
				result = std::bit_cast<uint32_t>(static_cast<float>(static_cast<uint32_t>(a)));
				return true;
			}
			return false;
		case ValueOpcode::ConvertU32F32:
			if (Arg(inst, 0, a)) {
				const auto value = Float32(a);
				if (!std::isfinite(value) || value < 0.0f ||
				    static_cast<double>(value) > UINT32_MAX) {
					return false;
				}
				result = static_cast<uint32_t>(value);
				return true;
			}
			return false;
		case ValueOpcode::FPMul32:
			if (binary()) {
				result = std::bit_cast<uint32_t>(Float32(a) * Float32(b));
				return true;
			}
			return false;
		case ValueOpcode::FPTrunc32:
			if (Arg(inst, 0, a)) {
				result = std::bit_cast<uint32_t>(std::trunc(Float32(a)));
				return true;
			}
			return false;
		case ValueOpcode::FPIsNan32:
			if (Arg(inst, 0, a)) {
				result = std::isnan(Float32(a));
				return true;
			}
			return false;
		case ValueOpcode::FPOrdLessThanEqual32:
			if (binary()) {
				result = Float32(a) <= Float32(b);
				return true;
			}
			return false;
		case ValueOpcode::FPOrdGreaterThanEqual32:
			if (binary()) {
				result = Float32(a) >= Float32(b);
				return true;
			}
			return false;
		case ValueOpcode::BitwiseAnd32:
			if (binary()) {
				result = static_cast<uint32_t>(a & b);
				return true;
			}
			return false;
		case ValueOpcode::BitwiseAnd64:
			if (binary()) {
				result = a & b;
				return true;
			}
			return false;
		case ValueOpcode::BitwiseOr32:
			if (binary()) {
				result = static_cast<uint32_t>(a | b);
				return true;
			}
			return false;
		case ValueOpcode::BitwiseXor32:
			if (binary()) {
				result = static_cast<uint32_t>(a ^ b);
				return true;
			}
			return false;
		case ValueOpcode::BitwiseNot32:
			if (Arg(inst, 0, a)) {
				result = ~static_cast<uint32_t>(a);
				return true;
			}
			return false;
		case ValueOpcode::ShiftLeftLogical32:
			if (binary()) {
				result = static_cast<uint32_t>(a) << (b & 31u);
				return true;
			}
			return false;
		case ValueOpcode::ShiftLeftLogical64:
			if (binary()) {
				result = a << (b & 63u);
				return true;
			}
			return false;
		case ValueOpcode::ShiftRightLogical32:
			if (binary()) {
				result = static_cast<uint32_t>(a) >> (b & 31u);
				return true;
			}
			return false;
		case ValueOpcode::ShiftRightLogical64:
			if (binary()) {
				result = a >> (b & 63u);
				return true;
			}
			return false;
		case ValueOpcode::ShiftRightArithmetic32:
			if (binary()) {
				result = static_cast<uint32_t>(
				    std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >> (b & 31u));
				return true;
			}
			return false;
		case ValueOpcode::ShiftRightArithmetic64:
			if (binary()) {
				result = static_cast<uint64_t>(std::bit_cast<int64_t>(a) >> (b & 63u));
				return true;
			}
			return false;
		case ValueOpcode::BitFieldUExtract:
			if (ternary()) {
				const auto offset = static_cast<uint32_t>(b);
				const auto width  = static_cast<uint32_t>(c);
				if (offset > 32u || width > 32u - offset) {
					return false;
				}
				const auto mask = width == 32u  ? UINT32_MAX
				                  : width == 0u ? 0u
				                                : (uint32_t {1} << width) - 1u;
				result = width == 0u ? 0u : (static_cast<uint32_t>(a) >> offset) & mask;
				return true;
			}
			return false;
		case ValueOpcode::BitFieldSExtract:
			if (ternary()) {
				const auto offset = static_cast<uint32_t>(b);
				const auto width  = static_cast<uint32_t>(c);
				if (offset > 32u || width > 32u - offset) {
					return false;
				}
				if (width == 0u) {
					result = 0;
					return true;
				}
				const auto mask = width == 32u ? UINT32_MAX : (uint32_t {1} << width) - 1u;
				auto       bits = (static_cast<uint32_t>(a) >> offset) & mask;
				if (width < 32u && (bits & (uint32_t {1} << (width - 1u))) != 0u) {
					bits |= ~mask;
				}
				result = bits;
				return true;
			}
			return false;
		case ValueOpcode::BitFieldInsert: {
			uint64_t d = 0;
			if (!ternary() || !Arg(inst, 3, d)) {
				return false;
			}
			const auto offset = static_cast<uint32_t>(c);
			const auto width  = static_cast<uint32_t>(d);
			if (offset > 32u || width > 32u - offset) {
				return false;
			}
			if (width == 0u) {
				result = static_cast<uint32_t>(a);
				return true;
			}
			const auto mask =
			    width == 32u ? UINT32_MAX : ((uint32_t {1} << width) - 1u) << offset;
			result = (static_cast<uint32_t>(a) & ~mask) |
			         ((static_cast<uint32_t>(b) << offset) & mask);
			return true;
		}
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectF32: {
			auto& predicate = m_clean_evaluator != nullptr ? *m_clean_evaluator : *this;
			if (predicate.EvaluateWide(inst.Arg(0), a)) {
				return Arg(inst, a != 0u ? 1u : 2u, result);
			}
			return false;
		}
		case ValueOpcode::IEqual32:
			if (binary()) {
				result = static_cast<uint32_t>(a) == static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::INotEqual32:
			if (binary()) {
				result = static_cast<uint32_t>(a) != static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::ULessThan32:
			if (binary()) {
				result = static_cast<uint32_t>(a) < static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::UGreaterThan32:
			if (binary()) {
				result = static_cast<uint32_t>(a) > static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::SGreaterThanEqual32:
			if (binary()) {
				result = std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >=
				         std::bit_cast<int32_t>(static_cast<uint32_t>(b));
				return true;
			}
			return false;
		case ValueOpcode::LogicalAnd:
			if (binary()) {
				result = (a != 0u) && (b != 0u);
				return true;
			}
			return false;
		case ValueOpcode::LogicalOr:
			if (binary()) {
				result = (a != 0u) || (b != 0u);
				return true;
			}
			return false;
		case ValueOpcode::LogicalXor:
			if (binary()) {
				result = (a != 0u) != (b != 0u);
				return true;
			}
			return false;
		case ValueOpcode::LogicalNot:
			if (Arg(inst, 0, a)) {
				result = a == 0u;
				return true;
			}
			return false;
		case ValueOpcode::UndefU1:
		case ValueOpcode::UndefU8:
		case ValueOpcode::UndefU16:
		case ValueOpcode::UndefU32:
		case ValueOpcode::UndefU64: return false;
		default: break;
	}
	return false;
}
bool SrtWalker::EvaluateDescriptor(uint32_t source, DescriptorValue& result) {
	if (source >= m_program.descriptor_sources.size()) {
		return false;
	}
	const auto& descriptor = m_program.descriptor_sources[source];
	result = {};
	result.dword_count = descriptor.dword_count;
	for (uint32_t index = 0; index < descriptor.dword_count; ++index) {
		if (!Evaluate(descriptor.dwords[index], result.dwords[index])) {
			return false;
		}
	}
	return true;
}

std::span<const uint8_t> SrtWalker::FindActiveSources() {
	if (m_program.control_flow.empty()) {
		return {};
	}
	// Conditions over flat SRT slots use this walker; see CompiledResourcePlan.
	const auto& direct = CompileResourcePlan(m_program).direct_conditions;
	auto&       strict = m_clean_evaluator != nullptr ? *m_clean_evaluator : *this;
	auto&       active = m_program.active_sources;
	active.assign(m_program.descriptor_sources.size(), 1u);
	for (const auto& block: m_program.control_flow) {
		for (const auto source: block.sources) {
			active.at(source) = 0u;
		}
	}
	auto& visited = m_program.visited_blocks;
	auto& pending = m_program.pending_blocks;
	visited.assign(m_program.control_flow.size(), 0u);
	pending.clear();
	pending.push_back(0u);
	while (!pending.empty()) {
		const auto index = pending.back();
		pending.pop_back();
		if (visited.at(index)) {
			continue;
		}
		visited[index] = 1u;
		const auto& block = m_program.control_flow[index];
		for (const auto source: block.sources) {
			active[source] = 1u;
		}
		uint32_t condition = 0;
		auto&    evaluator = index < direct.size() && direct[index] != 0u ? *this : strict;
		if (!block.condition.IsEmpty() && m_runtime.read_specialization_memory != nullptr &&
		    evaluator.Evaluate(block.condition, condition)) {
			pending.push_back(block.successors[condition != 0u ? 0u : 1u]);
		} else {
			pending.insert(pending.end(), block.successors.begin(), block.successors.end());
		}
	}
	return active;
}

bool SrtWalker::RefreshFlatBuffer(std::vector<uint32_t>& flat) {
	if (!m_program.srt_plan_complete) {
		return false;
	}
	flat.resize(m_program.srt_reads.size());
	for (const auto& read: m_program.srt_reads) {
		const bool clean = read.flat_offset < m_clean_flat_slots.size() &&
		                   m_clean_flat_slots[read.flat_offset] != 0u;
		if (clean && (m_clean_evaluator == nullptr || m_runtime.read_specialization_memory == nullptr)) {
			return false;
		}
		auto& evaluator = clean ? *m_clean_evaluator : *this;
		if (read.flat_offset >= flat.size() || !evaluator.Evaluate(read.value, flat[read.flat_offset])) {
			return false;
		}
	}
	return true;
}

namespace {

using NodeOp = ResourceNode::Op;

bool ImmediateBits(Value value, uint64_t& bits) {
	switch (value.GetType()) {
		case Type::U1: bits = value.U1(); return true;
		case Type::U8: bits = value.U8(); return true;
		case Type::U16: bits = value.U16(); return true;
		case Type::U32: bits = value.U32(); return true;
		case Type::U64: bits = value.U64(); return true;
		case Type::F32: bits = std::bit_cast<uint32_t>(value.F32Value()); return true;
		case Type::F16: bits = value.F16Bits(); return false;
		case Type::ScalarReg: bits = RegIndex(value.ScalarRegister()); return false;
		case Type::VectorReg: bits = RegIndex(value.VectorRegister()); return false;
		default: bits = 0; return false;
	}
}

NodeOp ArithmeticOp(ValueOpcode op, size_t& args) {
	args = 2;
	switch (op) {
		case ValueOpcode::CompositeConstructU64: return NodeOp::ConstructU64;
		case ValueOpcode::IAdd32: return NodeOp::IAdd32;
		case ValueOpcode::IAdd64: return NodeOp::IAdd64;
		case ValueOpcode::ISub32: return NodeOp::ISub32;
		case ValueOpcode::ISub64: return NodeOp::ISub64;
		case ValueOpcode::IMul32: return NodeOp::IMul32;
		case ValueOpcode::IMul64: return NodeOp::IMul64;
		case ValueOpcode::UMin32: return NodeOp::UMin32;
		case ValueOpcode::FPMul32: return NodeOp::FPMul32;
		case ValueOpcode::FPOrdLessThanEqual32: return NodeOp::FPOrdLessThanEqual32;
		case ValueOpcode::FPOrdGreaterThanEqual32: return NodeOp::FPOrdGreaterThanEqual32;
		case ValueOpcode::BitwiseAnd32: return NodeOp::BitwiseAnd32;
		case ValueOpcode::BitwiseAnd64: return NodeOp::BitwiseAnd64;
		case ValueOpcode::BitwiseOr32: return NodeOp::BitwiseOr32;
		case ValueOpcode::BitwiseXor32: return NodeOp::BitwiseXor32;
		case ValueOpcode::ShiftLeftLogical32: return NodeOp::ShiftLeftLogical32;
		case ValueOpcode::ShiftLeftLogical64: return NodeOp::ShiftLeftLogical64;
		case ValueOpcode::ShiftRightLogical32: return NodeOp::ShiftRightLogical32;
		case ValueOpcode::ShiftRightLogical64: return NodeOp::ShiftRightLogical64;
		case ValueOpcode::ShiftRightArithmetic32: return NodeOp::ShiftRightArithmetic32;
		case ValueOpcode::ShiftRightArithmetic64: return NodeOp::ShiftRightArithmetic64;
		case ValueOpcode::IEqual32: return NodeOp::IEqual32;
		case ValueOpcode::INotEqual32: return NodeOp::INotEqual32;
		case ValueOpcode::ULessThan32: return NodeOp::ULessThan32;
		case ValueOpcode::UGreaterThan32: return NodeOp::UGreaterThan32;
		case ValueOpcode::SGreaterThanEqual32: return NodeOp::SGreaterThanEqual32;
		case ValueOpcode::LogicalAnd: return NodeOp::LogicalAnd;
		case ValueOpcode::LogicalOr: return NodeOp::LogicalOr;
		case ValueOpcode::LogicalXor: return NodeOp::LogicalXor;
		default: break;
	}
	args = 1;
	switch (op) {
		case ValueOpcode::ConvertF32U32: return NodeOp::ConvertF32U32;
		case ValueOpcode::ConvertU32F32: return NodeOp::ConvertU32F32;
		case ValueOpcode::FPTrunc32: return NodeOp::FPTrunc32;
		case ValueOpcode::FPIsNan32: return NodeOp::FPIsNan32;
		case ValueOpcode::BitwiseNot32: return NodeOp::BitwiseNot32;
		case ValueOpcode::LogicalNot: return NodeOp::LogicalNot;
		default: break;
	}
	args = 3;
	switch (op) {
		case ValueOpcode::BitFieldUExtract: return NodeOp::BitFieldUExtract;
		case ValueOpcode::BitFieldSExtract: return NodeOp::BitFieldSExtract;
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectF32: return NodeOp::Select;
		default: break;
	}
	args = 4;
	return op == ValueOpcode::BitFieldInsert ? NodeOp::BitFieldInsert : NodeOp::Fail;
}

class PlanCompiler {
public:
	PlanCompiler(const ResourcePlan& program, CompiledResourcePlan& compiled)
	    : m_program(program), m_compiled(compiled) {}

	uint32_t Node(Value value) {
		value = value.Resolve();
		if (value.IsImmediate()) {
			uint64_t   bits  = 0;
			const bool valid = ImmediateBits(value, bits);
			const auto key   = std::pair {static_cast<uint32_t>(value.GetType()), bits};
			if (const auto found = m_constants.find(key); found != m_constants.end()) {
				return found->second;
			}
			const auto index = Push({.op = valid ? NodeOp::Const : NodeOp::Fail, .imm = bits});
			m_constants.emplace(key, index);
			return index;
		}
		const auto* inst = value.TryInstruction();
		if (const auto found = m_instructions.find(inst); found != m_instructions.end()) {
			return found->second;
		}
		// Register before the operands so that cycles resolve to this node, like the walker's
		// in-progress marker.
		const auto index = Push({});
		m_instructions.emplace(inst, index);
		const auto node        = Build(*inst);
		m_compiled.nodes[index] = node;
		return index;
	}

private:
	uint32_t Push(const ResourceNode& node) {
		m_compiled.nodes.push_back(node);
		return static_cast<uint32_t>(m_compiled.nodes.size() - 1u);
	}

	ResourceNode Build(const Inst& inst) {
		ResourceNode node;
		const auto   op = inst.GetOpcode();
		switch (op) {
			case ValueOpcode::GetUserData: {
				if (inst.NumArgs() < 1 || inst.Arg(0).GetType() != Type::ScalarReg) {
					return node;
				}
				const auto reg = RegIndex(inst.Arg(0).ScalarRegister());
				if (reg < m_program.user_data_base) {
					return node;
				}
				node.op  = NodeOp::UserData;
				node.aux = reg - m_program.user_data_base;
				return node;
			}
			case ValueOpcode::GetShaderBase: node.op = NodeOp::ShaderBase; return node;
			case ValueOpcode::Phi: {
				const auto invariant =
				    ResolveInvariantPhi(m_program, Value(const_cast<Inst*>(&inst)));
				if (invariant.IsEmpty()) {
					return node;
				}
				node.args[0] = Node(invariant);
				node.op      = NodeOp::Forward;
				return node;
			}
			case ValueOpcode::ReadFirstLane: {
				if (inst.NumArgs() < 2) {
					return node;
				}
				node.args[0]    = Node(inst.Arg(0));
				const auto mask = inst.Arg(1).Resolve();
				node.args[1]    = mask.IsEmpty() ? ResourceNode::NoNode : Node(mask);
				node.op         = NodeOp::ReadFirstLane;
				return node;
			}
			case ValueOpcode::BitCastU32F32:
			case ValueOpcode::BitCastF32U32:
				if (inst.NumArgs() < 1) {
					return node;
				}
				node.args[0] = Node(inst.Arg(0));
				node.op      = NodeOp::Forward;
				return node;
			case ValueOpcode::CompositeExtractU64:
			case ValueOpcode::CompositeExtractU32x2: return BuildExtract(inst);
			case ValueOpcode::ReadConst: {
				if (inst.NumArgs() < 2) {
					return node;
				}
				const auto slot = inst.Arg(1).Resolve();
				if (!slot.IsImmediate() || slot.GetType() != Type::U32 ||
				    slot.U32() >= m_program.srt_reads.size()) {
					return node;
				}
				node.aux     = slot.U32();
				node.args[0] = Node(m_program.srt_reads[slot.U32()].value);
				node.flags   = node.aux < m_program.clean_flat_slots.size() &&
				                     m_program.clean_flat_slots[node.aux] != 0u
				                   ? ResourceNode::CleanSlot
				                   : 0u;
				node.op      = NodeOp::ReadConst;
				return node;
			}
			case ValueOpcode::LoadAddressU32:
			case ValueOpcode::ReadConstBuffer: return BuildRawRead(inst);
			default: break;
		}
		size_t     count      = 0;
		const auto arithmetic = ArithmeticOp(op, count);
		if (arithmetic == NodeOp::Fail || inst.NumArgs() < count) {
			return node;
		}
		for (size_t index = 0; index < count; index++) {
			node.args[index] = Node(inst.Arg(index));
		}
		node.op = arithmetic;
		return node;
	}

	ResourceNode BuildExtract(const Inst& inst) {
		ResourceNode node;
		if (inst.NumArgs() < 2) {
			return node;
		}
		const auto index = inst.Arg(1).Resolve();
		if (!index.IsImmediate() || index.GetType() != Type::U32 || index.U32() >= 2u) {
			return node;
		}
		node.aux = index.U32();
		if (inst.GetOpcode() == ValueOpcode::CompositeExtractU64) {
			node.args[0] = Node(inst.Arg(0));
			node.op      = NodeOp::ExtractU64;
			return node;
		}
		const auto* source = inst.Arg(0).Resolve().TryInstruction();
		if (source == nullptr) {
			return node;
		}
		if (source->GetOpcode() == ValueOpcode::CompositeConstructU32x2 &&
		    source->NumArgs() > node.aux) {
			node.args[0] = Node(source->Arg(node.aux));
			node.op      = NodeOp::Forward;
		} else if (source->GetOpcode() == ValueOpcode::IAddCarry32 && source->NumArgs() >= 2) {
			node.args[0] = Node(source->Arg(0));
			node.args[1] = Node(source->Arg(1));
			node.op      = NodeOp::AddCarry;
		}
		return node;
	}

	ResourceNode BuildRawRead(const Inst& inst) {
		ResourceNode node;
		if (!IsRawRead(m_program, inst) || inst.NumArgs() < 2) {
			return node;
		}
		const auto* handle = inst.Arg(0).Resolve().TryInstruction();
		if (handle == nullptr || handle->NumArgs() < 2) {
			return node;
		}
		const auto& memory = m_program.memory_info[inst.Flags<MemoryFlags>().index];
		node.imm = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(memory.offset)));
		node.args[0] = Node(handle->Arg(0));
		node.args[1] = Node(handle->Arg(1));
		node.args[2] = Node(inst.Arg(1));
		if (inst.GetOpcode() == ValueOpcode::ReadConstBuffer) {
			if (handle->NumArgs() == 4u) {
				node.args[3] = Node(handle->Arg(2));
				node.args[4] = Node(handle->Arg(3));
			}
			node.op = NodeOp::RawBuffer;
		} else {
			node.op = NodeOp::RawAddress;
		}
		return node;
	}

	const ResourcePlan&                          m_program;
	CompiledResourcePlan&                        m_compiled;
	std::unordered_map<const Inst*, uint32_t>    m_instructions;
	std::map<std::pair<uint32_t, uint64_t>, uint32_t> m_constants;
};

// Finds the inputs of the buffer, image and sampler descriptor values. ReadConst is an input
// (its value is the flat SRT slot); memory reads other than a slot's own raw read, clean-only
// dependencies and nested EXEC evaluations make a plan non-memoizable.
class MemoAnalysis {
public:
	explicit MemoAnalysis(CompiledResourcePlan& compiled): m_compiled(compiled) {}

	bool Visit(uint32_t index, bool clean) {
		if (index == ResourceNode::NoNode) {
			return true;
		}
		auto& visited = clean ? m_clean_visited : m_visited;
		if (visited.size() < m_compiled.nodes.size()) {
			visited.resize(m_compiled.nodes.size(), 0u);
		}
		if (visited[index] != 0u) {
			return true;
		}
		visited[index] = 1u;
		const auto& node = m_compiled.nodes[index];
		switch (node.op) {
			case NodeOp::Fail:
			case NodeOp::Const: return true;
			case NodeOp::UserData: m_compiled.memo_user_data.push_back(node.aux); return true;
			case NodeOp::ShaderBase: m_compiled.memo_shader_base = true; return true;
			case NodeOp::ReadConst: {
				// A clean walker rereads the slot through the strict reader.
				const auto target = m_compiled.nodes[node.args[0]].op;
				if (clean || (node.flags & ResourceNode::CleanSlot) != 0u ||
				    (target != NodeOp::RawAddress && target != NodeOp::RawBuffer)) {
					return false;
				}
				m_compiled.memo_slots.push_back(node.aux);
				return true;
			}
			case NodeOp::ReadFirstLane:
			case NodeOp::RawAddress:
			case NodeOp::RawBuffer: return false;
			case NodeOp::Select:
				// Predicates are evaluated by the clean walker.
				return Visit(node.args[0], true) && Visit(node.args[1], clean) &&
				       Visit(node.args[2], clean);
			default:
				for (const auto arg: node.args) {
					if (!Visit(arg, clean)) {
						return false;
					}
				}
				return true;
		}
	}

private:
	CompiledResourcePlan& m_compiled;
	std::vector<uint8_t>  m_visited;
	std::vector<uint8_t>  m_clean_visited;
};

// A condition may use the ordinary walker when everything it reads directly is a flat SRT slot,
// which RefreshFlatBuffer reads (and memoizes) on every refresh anyway. Select predicates still go
// to the strict walker, as for any ordinary value.
class DirectConditionAnalysis {
public:
	explicit DirectConditionAnalysis(const CompiledResourcePlan& compiled)
	    : m_compiled(compiled), m_visited(compiled.nodes.size(), 0u) {}

	bool Visit(uint32_t index) {
		if (index == ResourceNode::NoNode || m_visited[index] != 0u) {
			return true;
		}
		m_visited[index] = 1u;
		const auto& node = m_compiled.nodes[index];
		switch (node.op) {
			case NodeOp::ReadConst: {
				const auto target = m_compiled.nodes[node.args[0]].op;
				return (node.flags & ResourceNode::CleanSlot) == 0u &&
				       (target == NodeOp::RawAddress || target == NodeOp::RawBuffer);
			}
			case NodeOp::ReadFirstLane:
			case NodeOp::RawAddress:
			case NodeOp::RawBuffer: return false;
			case NodeOp::Select: return Visit(node.args[1]) && Visit(node.args[2]);
			default:
				for (const auto arg: node.args) {
					if (!Visit(arg)) {
						return false;
					}
				}
				return true;
		}
	}

private:
	const CompiledResourcePlan& m_compiled;
	std::vector<uint8_t>        m_visited;
};

void AnalyzeControlFlow(const ResourcePlan& program, CompiledResourcePlan& compiled) {
	compiled.direct_conditions.assign(program.control_flow.size(), 0u);
	const bool clean_slots =
	    std::ranges::any_of(program.clean_flat_slots, [](uint8_t slot) { return slot != 0u; });
	for (uint32_t block = 0; block < program.control_flow.size(); block++) {
		const auto condition = compiled.conditions[block];
		compiled.direct_conditions[block] =
		    !clean_slots && condition != ResourceNode::NoNode &&
		    DirectConditionAnalysis(compiled).Visit(condition);
	}
	compiled.initial_active.assign(program.descriptor_sources.size(), 1u);
	for (const auto& block: program.control_flow) {
		for (const auto source: block.sources) {
			if (source >= compiled.initial_active.size()) {
				compiled.initial_active.clear();
				return;
			}
			compiled.initial_active[source] = 0u;
		}
	}
	// A block's successors are inert when no block reachable from them (along any path) has
	// sources: whatever the condition, the walk adds nothing.
	const auto count = program.control_flow.size();
	compiled.inert_successors.assign(count, 0u);
	std::vector<uint8_t>  reached;
	std::vector<uint32_t> stack;
	for (uint32_t block = 0; block < count; block++) {
		reached.assign(count, 0u);
		stack.assign(program.control_flow[block].successors.begin(),
		             program.control_flow[block].successors.end());
		bool inert = true;
		while (!stack.empty() && inert) {
			const auto index = stack.back();
			stack.pop_back();
			if (index >= count) {
				inert = false;
				break;
			}
			if (reached[index] != 0u) {
				continue;
			}
			reached[index] = 1u;
			inert          = program.control_flow[index].sources.empty();
			stack.insert(stack.end(), program.control_flow[index].successors.begin(),
			             program.control_flow[index].successors.end());
		}
		compiled.inert_successors[block] = inert ? 1u : 0u;
	}
}

void AnalyzeMemo(const ResourcePlan& program, CompiledResourcePlan& compiled) {
	if (program.requires_specialization_memory ||
	    std::ranges::any_of(program.clean_flat_slots, [](uint8_t slot) { return slot != 0u; })) {
		return;
	}
	MemoAnalysis analysis(compiled);
	const auto   visit_source = [&](uint32_t source) {
        if (source >= compiled.descriptors.size()) {
            return true;
        }
        const auto count = program.descriptor_sources[source].dword_count;
        for (uint32_t dword = 0; dword < count && dword < 8u; dword++) {
            if (!analysis.Visit(compiled.descriptors[source][dword], false)) {
                return false;
            }
        }
        return true;
	};
	for (const auto& buffer: program.info.buffers) {
		if (!visit_source(buffer.source)) return;
	}
	for (const auto& image: program.info.images) {
		if (!visit_source(image.source)) return;
	}
	for (const auto& sampler: program.info.samplers) {
		if (!visit_source(sampler.source)) return;
	}
	for (auto& slot: compiled.memo_slots) {
		slot = program.srt_reads[slot].flat_offset;
		if (slot >= program.srt_reads.size()) return;
	}
	for (auto* list: {&compiled.memo_user_data, &compiled.memo_slots}) {
		std::ranges::sort(*list);
		list->erase(std::unique(list->begin(), list->end()), list->end());
	}
	compiled.memoizable = true;
}

bool g_flat_run_reads = true; // See SetFlatRunReads.

// Groups the ordinary slots that read consecutive dwords through one handle and offset (a
// scalar load of several dwords becomes one LoadAddressU32 per dword) into runs.
void BuildFlatRuns(const ResourcePlan& program, CompiledResourcePlan& compiled) {
	if (!g_flat_run_reads) {
		return;
	}
	struct Candidate {
		uint32_t low    = 0;
		uint32_t high   = 0;
		uint32_t offset = 0;
		int64_t  imm    = 0;
		uint32_t slot   = 0;

		[[nodiscard]] bool SameBase(const Candidate& other) const {
			return low == other.low && high == other.high && offset == other.offset;
		}
	};
	std::vector<Candidate> candidates;
	for (uint32_t slot = 0; slot < compiled.slots.size(); slot++) {
		const auto flat_offset = program.srt_reads[slot].flat_offset;
		if (flat_offset >= program.srt_reads.size() ||
		    (flat_offset < compiled.clean_slots.size() && compiled.clean_slots[flat_offset] != 0u)) {
			continue;
		}
		const auto& node = compiled.nodes[compiled.slots[slot]];
		const auto  imm  = static_cast<int64_t>(node.imm);
		if (node.op != NodeOp::RawAddress || (imm & 3) != 0) {
			continue;
		}
		candidates.push_back({node.args[0], node.args[1], node.args[2], imm, slot});
	}
	std::ranges::sort(candidates, [](const Candidate& a, const Candidate& b) {
		return std::tie(a.low, a.high, a.offset, a.imm, a.slot) <
		       std::tie(b.low, b.high, b.offset, b.imm, b.slot);
	});
	std::vector<CompiledResourcePlan::FlatRunEntry> entries;
	const auto flush = [&](int64_t first_imm, int64_t last_imm) {
		// A single dword gains nothing from a run.
		if (entries.size() >= 2u && last_imm > first_imm) {
			compiled.flat_runs.push_back({
			    .first  = static_cast<uint32_t>(compiled.run_entries.size()),
			    .count  = static_cast<uint32_t>(entries.size()),
			    .dwords = static_cast<uint32_t>((last_imm - first_imm) / 4 + 1),
			});
			compiled.run_entries.insert(compiled.run_entries.end(), entries.begin(), entries.end());
		}
		entries.clear();
	};
	int64_t first_imm = 0;
	int64_t last_imm  = 0;
	for (size_t index = 0; index < candidates.size(); index++) {
		const auto& candidate = candidates[index];
		const bool  extends   = index != 0 && candidate.SameBase(candidates[index - 1]) &&
		                     candidate.imm - last_imm <= 4 &&
		                     (candidate.imm - first_imm) / 4 < CompiledResourcePlan::MaxRunDwords;
		if (!extends) {
			flush(first_imm, last_imm);
			first_imm = candidate.imm;
		}
		last_imm = candidate.imm;
		entries.push_back({candidate.slot, static_cast<uint32_t>((candidate.imm - first_imm) / 4),
		                   program.srt_reads[candidate.slot].flat_offset});
	}
	flush(first_imm, last_imm);
	if (!compiled.flat_runs.empty()) {
		compiled.in_run.assign(compiled.slots.size(), 0u);
		for (const auto& entry: compiled.run_entries) {
			compiled.in_run[entry.slot] = 1u;
		}
		// The first entry of a run reads dword 0, and its node gives the run's base.
		for (const auto& run: compiled.flat_runs) {
			EXIT_IF(compiled.run_entries[run.first].dword != 0u);
		}
	}
}

} // namespace

void SetFlatRunReads(bool enabled) {
	g_flat_run_reads = enabled;
}

const CompiledResourcePlan& CompileResourcePlan(const ResourcePlan& program) {
	if (program.compiled != nullptr) {
		return *program.compiled;
	}
	auto         compiled = std::make_unique<CompiledResourcePlan>();
	PlanCompiler compiler(program, *compiled);
	compiled->slots.reserve(program.srt_reads.size());
	for (const auto& read: program.srt_reads) {
		compiled->slots.push_back(compiler.Node(read.value));
	}
	compiled->clean_slots = program.clean_flat_slots;
	compiled->descriptors.resize(program.descriptor_sources.size());
	compiled->key_counts.resize(program.descriptor_sources.size(), ResourceNode::NoNode);
	compiled->selector_masks.resize(program.descriptor_sources.size(), ResourceNode::NoNode);
	for (uint32_t source = 0; source < program.descriptor_sources.size(); source++) {
		const auto& descriptor = program.descriptor_sources[source];
		auto&       dwords     = compiled->descriptors[source];
		dwords.fill(ResourceNode::NoNode);
		for (uint32_t dword = 0; dword < descriptor.dword_count && dword < dwords.size(); dword++) {
			dwords[dword] = compiler.Node(descriptor.dwords[dword]);
		}
		if (descriptor.indirect_image.has_value()) {
			compiled->key_counts[source] = compiler.Node(descriptor.indirect_image->key_count);
			compiled->selector_masks[source] =
			    compiler.Node(descriptor.indirect_image->selector_mask);
		}
	}
	compiled->conditions.reserve(program.control_flow.size());
	for (const auto& block: program.control_flow) {
		compiled->conditions.push_back(block.condition.IsEmpty() ? ResourceNode::NoNode
		                                                         : compiler.Node(block.condition));
	}
	for (uint32_t i = 0; i < program.uniform_fill.fill.words && i < compiled->fill.size(); i++) {
		compiled->fill[i] = compiler.Node(program.uniform_fill.values[i]);
	}
	compiled->descriptor_slots.resize(compiled->descriptors.size());
	for (uint32_t source = 0; source < compiled->descriptors.size(); source++) {
		for (uint32_t dword = 0; dword < 8u; dword++) {
			auto&      slot  = compiled->descriptor_slots[source][dword];
			const auto index = compiled->descriptors[source][dword];
			slot             = ResourceNode::NoNode;
			if (index == ResourceNode::NoNode) {
				continue;
			}
			const auto& node = compiled->nodes[index];
			if (node.op != NodeOp::ReadConst || (node.flags & ResourceNode::CleanSlot) != 0u) {
				continue;
			}
			const auto target = compiled->nodes[node.args[0]].op;
			if (target == NodeOp::RawAddress || target == NodeOp::RawBuffer) {
				slot = program.srt_reads[node.aux].flat_offset;
			}
		}
	}
	AnalyzeControlFlow(program, *compiled);
	AnalyzeMemo(program, *compiled);
	BuildFlatRuns(program, *compiled);
	program.compiled = std::move(compiled);
	return *program.compiled;
}

SrtEvaluator::SrtEvaluator(const ResourcePlan& program, const CompiledResourcePlan& compiled,
                           const SrtRuntime& runtime, bool clean_flat_slots,
                           SrtEvaluator* clean_evaluator, uint32_t active_mask)
    : m_program(program), m_compiled(compiled), m_nodes(compiled.nodes.data()),
      m_runtime(runtime), m_clean_flat_slots(clean_flat_slots),
      m_clean_evaluator(clean_evaluator), m_active_mask(active_mask),
      m_context(SrtWalker::AcquireContext(program)) {
	if (m_context.values.size() < compiled.nodes.size()) {
		m_context.values.resize(compiled.nodes.size());
	}
	m_memo       = m_context.values.data();
	m_generation = m_context.generation;
}

SrtEvaluator::~SrtEvaluator() { --m_program.evaluation_depth; }

bool SrtEvaluator::Evaluate(uint32_t node, uint32_t& result) {
	uint64_t wide = 0;
	if (!EvaluateWide(node, wide)) {
		return false;
	}
	result = static_cast<uint32_t>(wide);
	return true;
}

bool SrtEvaluator::EvaluateWide(uint32_t index, uint64_t& result) {
	const auto& node = m_nodes[index];
	if (node.op == NodeOp::Const) {
		result = node.imm;
		return true;
	}
	if (node.op == NodeOp::Fail) {
		return false;
	}
	if (m_active_mask != ResourceNode::NoNode && node.op == NodeOp::Select &&
	    node.args[0] == m_active_mask) {
		return EvaluateWide(node.args[1], result);
	}
	auto& memo = m_memo[index];
	if (memo.generation == m_generation) {
		result = memo.value;
		return true;
	}
	// The low generation bit marks an instruction that is still being evaluated.
	if (memo.generation == (m_generation | 1u)) {
		return false;
	}
	memo.generation = m_generation | 1u;
	uint64_t out    = 0;
	if (!EvaluateInst(node, out)) {
		memo.generation = 0;
		return false;
	}
	memo.value      = out;
	memo.generation = m_generation;
	result          = out;
	return true;
}

bool SrtEvaluator::EvaluateRawRead(const ResourceNode& node, uint64_t& result) {
	uint64_t low    = 0;
	uint64_t high   = 0;
	uint64_t offset = 0;
	if (!EvaluateWide(node.args[0], low) || !EvaluateWide(node.args[1], high) ||
	    !EvaluateWide(node.args[2], offset)) {
		return false;
	}
	const auto base      = ((high << 32u) | static_cast<uint32_t>(low)) & AddressMask;
	const auto immediate = static_cast<int64_t>(node.imm);
	uint64_t   address   = 0;
	if (node.op == NodeOp::RawBuffer) {
		uint64_t records = 0;
		uint64_t word3   = 0;
		if (node.args[3] == ResourceNode::NoNode || !EvaluateWide(node.args[3], records) ||
		    !EvaluateWide(node.args[4], word3)) {
			return false;
		}
		if (immediate < 0) {
			return false;
		}
		const auto byte_offset =
		    static_cast<uint64_t>(immediate) + static_cast<uint32_t>(offset);
		const auto aligned = byte_offset & ~uint64_t {3};
		const auto stride  = (static_cast<uint32_t>(high) >> 16u) & 0x3fffu;
		const auto size = stride == 0u
		                      ? static_cast<uint64_t>(static_cast<uint32_t>(records))
		                      : static_cast<uint64_t>(stride) * static_cast<uint32_t>(records);
		if (aligned > size || size - aligned < sizeof(uint32_t)) {
			return false;
		}
		address = ((base & ~uint64_t {3}) + byte_offset) & ~uint64_t {3};
	} else {
		const auto relative = (immediate & ~int64_t {3}) +
		                      static_cast<int64_t>(static_cast<uint32_t>(offset) & ~3u);
		if (!AddSignedAddress(base & ~uint64_t {3}, relative, address)) {
			return false;
		}
	}
	uint32_t word = 0;
	if (m_runtime.read_memory != nullptr) {
		if (!m_runtime.read_memory(m_runtime.userdata, address, {&word, 1})) {
			return false;
		}
	} else {
		std::memcpy(&word, reinterpret_cast<const void*>(address), sizeof(word));
	}
	result = word;
	return true;
}

bool SrtEvaluator::EvaluateInst(const ResourceNode& node, uint64_t& result) {
	uint64_t   a       = 0;
	uint64_t   b       = 0;
	uint64_t   c       = 0;
	const auto unary   = [&]() { return EvaluateWide(node.args[0], a); };
	const auto binary  = [&]() {
        return EvaluateWide(node.args[0], a) && EvaluateWide(node.args[1], b);
	};
	const auto ternary = [&]() { return binary() && EvaluateWide(node.args[2], c); };
	const auto f32     = [](uint64_t bits) {
        return std::bit_cast<float>(static_cast<uint32_t>(bits));
	};
	switch (node.op) {
		case NodeOp::UserData:
			if (node.aux >= m_runtime.user_data.size()) {
				return false;
			}
			result = m_runtime.user_data[node.aux];
			return true;
		case NodeOp::ShaderBase: result = m_runtime.shader_base; return true;
		case NodeOp::Forward: return EvaluateWide(node.args[0], result);
		case NodeOp::ReadConst:
			// After RefreshFlatBuffer the slot's value is in the flat buffer, read as this read
			// would evaluate it (clean slots by the clean walker).
			if (m_flat != nullptr && node.aux < m_program.srt_reads.size()) {
				result = (*m_flat)[m_program.srt_reads[node.aux].flat_offset];
				return true;
			}
			if ((node.flags & ResourceNode::CleanSlot) != 0u && m_clean_flat_slots &&
			    m_clean_evaluator != nullptr) {
				return m_clean_evaluator->EvaluateWide(node.args[0], result);
			}
			return EvaluateWide(node.args[0], result);
		case NodeOp::ReadFirstLane: {
			const auto   clean_runtime = CleanRuntime(m_runtime);
			SrtEvaluator clean_active(m_program, m_compiled, clean_runtime, false, nullptr,
			                          node.args[1]);
			SrtEvaluator active(m_program, m_compiled, m_runtime, m_clean_flat_slots,
			                    &clean_active, node.args[1]);
			return active.EvaluateWide(node.args[0], result);
		}
		case NodeOp::RawAddress:
		case NodeOp::RawBuffer: return EvaluateRawRead(node, result);
		case NodeOp::ExtractU64:
			if (!unary()) {
				return false;
			}
			result = static_cast<uint32_t>(a >> (node.aux * 32u));
			return true;
		case NodeOp::AddCarry: {
			if (!binary()) {
				return false;
			}
			const auto sum =
			    static_cast<uint64_t>(static_cast<uint32_t>(a)) + static_cast<uint32_t>(b);
			result = node.aux == 0u ? static_cast<uint32_t>(sum)
			                        : static_cast<uint32_t>(sum >> 32u);
			return true;
		}
		case NodeOp::ConstructU64:
			if (!binary()) {
				return false;
			}
			result = static_cast<uint32_t>(a) |
			         (static_cast<uint64_t>(static_cast<uint32_t>(b)) << 32u);
			return true;
		case NodeOp::IAdd32:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a + b);
			return true;
		case NodeOp::IAdd64:
			if (!binary()) return false;
			result = a + b;
			return true;
		case NodeOp::ISub32:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a - b);
			return true;
		case NodeOp::ISub64:
			if (!binary()) return false;
			result = a - b;
			return true;
		case NodeOp::IMul32:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a * b);
			return true;
		case NodeOp::IMul64:
			if (!binary()) return false;
			result = a * b;
			return true;
		case NodeOp::UMin32:
			if (!binary()) return false;
			result = std::min(static_cast<uint32_t>(a), static_cast<uint32_t>(b));
			return true;
		case NodeOp::ConvertF32U32:
			if (!unary()) return false;
			result = std::bit_cast<uint32_t>(static_cast<float>(static_cast<uint32_t>(a)));
			return true;
		case NodeOp::ConvertU32F32: {
			if (!unary()) return false;
			const auto value = f32(a);
			if (!std::isfinite(value) || value < 0.0f || static_cast<double>(value) > UINT32_MAX) {
				return false;
			}
			result = static_cast<uint32_t>(value);
			return true;
		}
		case NodeOp::FPMul32:
			if (!binary()) return false;
			result = std::bit_cast<uint32_t>(f32(a) * f32(b));
			return true;
		case NodeOp::FPTrunc32:
			if (!unary()) return false;
			result = std::bit_cast<uint32_t>(std::trunc(f32(a)));
			return true;
		case NodeOp::FPIsNan32:
			if (!unary()) return false;
			result = std::isnan(f32(a));
			return true;
		case NodeOp::FPOrdLessThanEqual32:
			if (!binary()) return false;
			result = f32(a) <= f32(b);
			return true;
		case NodeOp::FPOrdGreaterThanEqual32:
			if (!binary()) return false;
			result = f32(a) >= f32(b);
			return true;
		case NodeOp::BitwiseAnd32:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a & b);
			return true;
		case NodeOp::BitwiseAnd64:
			if (!binary()) return false;
			result = a & b;
			return true;
		case NodeOp::BitwiseOr32:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a | b);
			return true;
		case NodeOp::BitwiseXor32:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a ^ b);
			return true;
		case NodeOp::BitwiseNot32:
			if (!unary()) return false;
			result = ~static_cast<uint32_t>(a);
			return true;
		case NodeOp::ShiftLeftLogical32:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a) << (b & 31u);
			return true;
		case NodeOp::ShiftLeftLogical64:
			if (!binary()) return false;
			result = a << (b & 63u);
			return true;
		case NodeOp::ShiftRightLogical32:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a) >> (b & 31u);
			return true;
		case NodeOp::ShiftRightLogical64:
			if (!binary()) return false;
			result = a >> (b & 63u);
			return true;
		case NodeOp::ShiftRightArithmetic32:
			if (!binary()) return false;
			result = static_cast<uint32_t>(std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >>
			                               (b & 31u));
			return true;
		case NodeOp::ShiftRightArithmetic64:
			if (!binary()) return false;
			result = static_cast<uint64_t>(std::bit_cast<int64_t>(a) >> (b & 63u));
			return true;
		case NodeOp::BitFieldUExtract: {
			if (!ternary()) return false;
			const auto offset = static_cast<uint32_t>(b);
			const auto width  = static_cast<uint32_t>(c);
			if (offset > 32u || width > 32u - offset) {
				return false;
			}
			const auto mask = width == 32u  ? UINT32_MAX
			                  : width == 0u ? 0u
			                                : (uint32_t {1} << width) - 1u;
			result = width == 0u ? 0u : (static_cast<uint32_t>(a) >> offset) & mask;
			return true;
		}
		case NodeOp::BitFieldSExtract: {
			if (!ternary()) return false;
			const auto offset = static_cast<uint32_t>(b);
			const auto width  = static_cast<uint32_t>(c);
			if (offset > 32u || width > 32u - offset) {
				return false;
			}
			if (width == 0u) {
				result = 0;
				return true;
			}
			const auto mask = width == 32u ? UINT32_MAX : (uint32_t {1} << width) - 1u;
			auto       bits = (static_cast<uint32_t>(a) >> offset) & mask;
			if (width < 32u && (bits & (uint32_t {1} << (width - 1u))) != 0u) {
				bits |= ~mask;
			}
			result = bits;
			return true;
		}
		case NodeOp::BitFieldInsert: {
			uint64_t d = 0;
			if (!ternary() || !EvaluateWide(node.args[3], d)) {
				return false;
			}
			const auto offset = static_cast<uint32_t>(c);
			const auto width  = static_cast<uint32_t>(d);
			if (offset > 32u || width > 32u - offset) {
				return false;
			}
			if (width == 0u) {
				result = static_cast<uint32_t>(a);
				return true;
			}
			const auto mask =
			    width == 32u ? UINT32_MAX : ((uint32_t {1} << width) - 1u) << offset;
			result = (static_cast<uint32_t>(a) & ~mask) |
			         ((static_cast<uint32_t>(b) << offset) & mask);
			return true;
		}
		case NodeOp::Select: {
			auto& predicate = m_clean_evaluator != nullptr ? *m_clean_evaluator : *this;
			if (!predicate.EvaluateWide(node.args[0], a)) {
				return false;
			}
			return EvaluateWide(node.args[a != 0u ? 1u : 2u], result);
		}
		case NodeOp::IEqual32:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a) == static_cast<uint32_t>(b);
			return true;
		case NodeOp::INotEqual32:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a) != static_cast<uint32_t>(b);
			return true;
		case NodeOp::ULessThan32:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a) < static_cast<uint32_t>(b);
			return true;
		case NodeOp::UGreaterThan32:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a) > static_cast<uint32_t>(b);
			return true;
		case NodeOp::SGreaterThanEqual32:
			if (!binary()) return false;
			result = std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >=
			         std::bit_cast<int32_t>(static_cast<uint32_t>(b));
			return true;
		case NodeOp::LogicalAnd:
			if (!binary()) return false;
			result = (a != 0u) && (b != 0u);
			return true;
		case NodeOp::LogicalOr:
			if (!binary()) return false;
			result = (a != 0u) || (b != 0u);
			return true;
		case NodeOp::LogicalXor:
			if (!binary()) return false;
			result = (a != 0u) != (b != 0u);
			return true;
		case NodeOp::LogicalNot:
			if (!unary()) return false;
			result = a == 0u;
			return true;
		case NodeOp::Fail:
		case NodeOp::Const: break;
	}
	return false;
}

bool SrtEvaluator::EvaluateDescriptor(uint32_t source, DescriptorValue& result) {
	if (source >= m_program.descriptor_sources.size()) {
		return false;
	}
	const auto& dwords = m_compiled.descriptors[source];
	const auto& slots  = m_compiled.descriptor_slots[source];
	result             = {};
	result.dword_count = m_program.descriptor_sources[source].dword_count;
	for (uint32_t index = 0; index < result.dword_count; ++index) {
		// A dword that is exactly a slot this walker refreshed evaluates to its flat value.
		if (m_flat != nullptr && slots[index] != ResourceNode::NoNode) {
			result.dwords[index] = (*m_flat)[slots[index]];
		} else if (!Evaluate(dwords[index], result.dwords[index])) {
			return false;
		}
	}
	return true;
}

std::span<const uint8_t> SrtEvaluator::FindActiveSources() {
	if (m_program.control_flow.empty()) {
		return {};
	}
	auto&      active    = m_program.active_sources;
	auto&      strict    = m_clean_evaluator != nullptr ? *m_clean_evaluator : *this;
	const bool evaluates = m_runtime.read_specialization_memory != nullptr;
	const auto outcome   = [&](uint32_t index) -> uint8_t {
		uint32_t condition = 0;
		auto&    evaluator = m_compiled.direct_conditions[index] != 0u ? *this : strict;
		if (!evaluator.Evaluate(m_compiled.conditions[index], condition)) {
			return 2u;
		}
		return condition != 0u ? 1u : 0u;
	};
	// The walk below depends on the plan only through the outcomes of the conditions it
	// evaluates, in an order those outcomes fix. Follow the tree of earlier walks (see
	// ResourcePlan::ActiveTreeNode) with this refresh's outcomes; reaching a leaf means the walk
	// would visit the same blocks and find the same sources.
	using TreeNode = ResourcePlan::ActiveTreeNode;
	auto& tree     = m_program.active_tree;
	auto& trace    = m_program.active_trace;
	if (evaluates && !tree.empty()) {
		uint32_t node = 0;
		while (node != TreeNode::None && tree[node].block != TreeNode::None) {
			node = tree[node].next[outcome(tree[node].block)];
		}
		if (node != TreeNode::None && tree[node].sources != TreeNode::None) {
			const auto first = m_program.active_tree_sources.begin() + tree[node].sources;
			const auto count = static_cast<ptrdiff_t>(m_program.descriptor_sources.size());
			active.assign(first, first + count);
			return active;
		}
	}
	trace.clear();
	if (!m_compiled.initial_active.empty()) {
		active.assign(m_compiled.initial_active.begin(), m_compiled.initial_active.end());
	} else {
		active.assign(m_program.descriptor_sources.size(), 1u);
		for (const auto& block: m_program.control_flow) {
			for (const auto source: block.sources) {
				active.at(source) = 0u;
			}
		}
	}
	auto&      visited     = m_program.visited_blocks;
	auto&      pending     = m_program.pending_blocks;
	const auto block_count = m_program.control_flow.size();
	// Most plans have at most 64 blocks; track those in a mask instead of clearing a vector.
	const bool small       = block_count <= 64u;
	uint64_t   visited_mask = 0;
	if (!small) {
		visited.assign(block_count, 0u);
	}
	pending.clear();
	pending.push_back(0u);
	while (!pending.empty()) {
		const auto index = pending.back();
		pending.pop_back();
		EXIT_IF(index >= block_count);
		if (small) {
			const auto bit = uint64_t {1} << index;
			if ((visited_mask & bit) != 0u) {
				continue;
			}
			visited_mask |= bit;
		} else {
			if (visited[index]) {
				continue;
			}
			visited[index] = 1u;
		}
		const auto& block = m_program.control_flow[index];
		for (const auto source: block.sources) {
			active[source] = 1u;
		}
		// Nothing reachable from here adds a source: skip the condition and its strict reads.
		if (!m_compiled.inert_successors.empty() && m_compiled.inert_successors[index] != 0u) {
			continue;
		}
		const auto condition_node = m_compiled.conditions[index];
		const auto result =
		    condition_node != ResourceNode::NoNode && evaluates ? outcome(index) : uint8_t {2};
		if (condition_node != ResourceNode::NoNode && evaluates) {
			trace.push_back({index, result});
		}
		if (result != 2u) {
			pending.push_back(block.successors[result != 0u ? 0u : 1u]);
		} else {
			pending.insert(pending.end(), block.successors.begin(), block.successors.end());
		}
	}
	// Add this walk to the tree, up to a bound on its size.
	constexpr size_t MaxTreeNodes = 4096;
	auto&            sources      = m_program.active_tree_sources;
	if (evaluates && active.size() == m_program.descriptor_sources.size() &&
	    tree.size() + trace.size() + 1u <= MaxTreeNodes) {
		if (tree.empty()) {
			tree.emplace_back();
		}
		uint32_t node = 0;
		bool     fits = true;
		for (const auto& step: trace) {
			// Earlier walks with the same outcomes so far evaluated the same condition here.
			if (tree[node].block == TreeNode::None && tree[node].sources == TreeNode::None) {
				tree[node].block = step.block;
			} else if (tree[node].block != step.block) {
				fits = false;
				break;
			}
			auto next = tree[node].next[step.outcome];
			if (next == TreeNode::None) {
				next                          = static_cast<uint32_t>(tree.size());
				tree[node].next[step.outcome] = next;
				tree.emplace_back();
			}
			node = next;
		}
		fits = fits && tree[node].block == TreeNode::None;
		if (fits && tree[node].sources == TreeNode::None) {
			tree[node].sources = static_cast<uint32_t>(sources.size());
			sources.insert(sources.end(), active.begin(), active.end());
		} else if (!fits) {
			// Not expected: the walk order is a function of the outcomes. Start over.
			tree.clear();
			sources.clear();
		}
	}
	return active;
}

bool SrtEvaluator::RefreshFlatBuffer(std::vector<uint32_t>& flat) {
	if (!m_program.srt_plan_complete) {
		return false;
	}
	flat.resize(m_program.srt_reads.size());
	// Run slots are ordinary: this walker evaluates them. A run that cannot be read whole
	// evaluates its slots one by one, and fails exactly as that would.
	const bool runs = !m_compiled.flat_runs.empty() && m_active_mask == ResourceNode::NoNode;
	for (uint32_t run = 0; runs && run < m_compiled.flat_runs.size(); run++) {
		const auto& info = m_compiled.flat_runs[run];
		if (ReadFlatRun(info, flat)) {
			continue;
		}
		for (uint32_t index = 0; index < info.count; index++) {
			const auto slot = m_compiled.run_entries[info.first + index].slot;
			if (!Evaluate(m_compiled.slots[slot], flat[m_program.srt_reads[slot].flat_offset])) {
				return false;
			}
		}
	}
	for (uint32_t slot = 0; slot < m_program.srt_reads.size(); slot++) {
		if (runs && m_compiled.in_run[slot] != 0u) {
			continue;
		}
		const auto offset = m_program.srt_reads[slot].flat_offset;
		const bool clean  = m_clean_flat_slots && offset < m_compiled.clean_slots.size() &&
		                   m_compiled.clean_slots[offset] != 0u;
		if (clean &&
		    (m_clean_evaluator == nullptr || m_runtime.read_specialization_memory == nullptr)) {
			return false;
		}
		auto& evaluator = clean ? *m_clean_evaluator : *this;
		if (offset >= flat.size() || !evaluator.Evaluate(m_compiled.slots[slot], flat[offset])) {
			return false;
		}
	}
	// Only this walker's own contexts use the shortcut; nested EXEC walkers evaluate normally.
	if (m_active_mask == ResourceNode::NoNode) {
		m_flat = &flat;
	}
	return true;
}

bool SrtEvaluator::ReadFlatRun(const CompiledResourcePlan::FlatRun& run,
                               std::vector<uint32_t>& flat) {
	const auto* entries = m_compiled.run_entries.data() + run.first;
	const auto& node    = m_nodes[m_compiled.slots[entries[0].slot]];
	uint64_t    low     = 0;
	uint64_t    high    = 0;
	uint64_t    offset  = 0;
	if (!EvaluateWide(node.args[0], low) || !EvaluateWide(node.args[1], high) ||
	    !EvaluateWide(node.args[2], offset)) {
		return false;
	}
	// As EvaluateRawRead computes each dword's address; the immediates are dword multiples.
	const auto base     = (((high << 32u) | static_cast<uint32_t>(low)) & AddressMask) & ~uint64_t {3};
	const auto relative = static_cast<int64_t>(node.imm) +
	                      static_cast<int64_t>(static_cast<uint32_t>(offset) & ~3u);
	const auto span     = static_cast<int64_t>(run.dwords - 1u) * 4;
	uint64_t   first    = 0;
	uint64_t   last     = 0;
	if (!AddSignedAddress(base, relative, first) || !AddSignedAddress(base, relative + span, last) ||
	    last - first != static_cast<uint64_t>(span)) {
		return false;
	}
	// One read per 64-byte block, as dword reads through a block cache would see memory.
	std::array<uint32_t, CompiledResourcePlan::MaxRunDwords> words;
	for (uint32_t done = 0; done < run.dwords;) {
		const auto address = first + uint64_t {done} * 4u;
		const auto count   = std::min<uint64_t>(run.dwords - done, (64u - (address & 63u)) / 4u);
		const auto values  = std::span(words.data() + done, count);
		if (m_runtime.read_memory != nullptr) {
			if (!m_runtime.read_memory(m_runtime.userdata, address, values)) {
				return false;
			}
		} else {
			std::memcpy(values.data(), reinterpret_cast<const void*>(address), values.size_bytes());
		}
		done += static_cast<uint32_t>(count);
	}
	// Later ReadConsts of these slots read the flat buffer (see EvaluateInst), so the slot nodes'
	// memos are left alone.
	for (uint32_t index = 0; index < run.count; index++) {
		flat[entries[index].flat_offset] = words[entries[index].dword];
	}
	return true;
}

bool ValidateRuntimeValue(const ResourcePlan& program, Value value, RuntimeValueType type) {
	return RuntimeValidator(program, type).Run(value);
}

void BuildSrtPlan(Program& program) {
	if (program.resource_tracking_complete) {
		EXIT("shader SRT planning failed: cannot rebuild SRT after resource tracking");
	}
	program.srt_plan_complete = false;
	PlanBuilder(program).Run();
	program.srt_plan_complete = true;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
