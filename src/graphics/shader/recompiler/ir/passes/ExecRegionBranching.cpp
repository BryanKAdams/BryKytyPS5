#include "graphics/shader/recompiler/ir/passes/ExecRegionBranching.h"

#include "graphics/shader/recompiler/ir/passes/ConstantPropagation.h"
#include "graphics/shader/recompiler/ir/passes/DeadCodeElimination.h"

#include <algorithm>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

Value Arg(const Inst& inst, size_t index) {
	return inst.Arg(index).Resolve();
}

const Inst* InstructionOf(Value value) {
	return value.Resolve().TryInstruction();
}

bool IsU32(Value value, uint32_t expected) {
	value = value.Resolve();
	return value.IsImmediate() && value.GetType() == Type::U32 && value.U32() == expected;
}

// The p of Ballot(p) that a mask is: the ballot's word 0, or its words 0 and 1 ORed (wave64).
std::optional<Value> BallotPredicate(Value mask) {
	const auto* inst = InstructionOf(mask);
	if (inst == nullptr) {
		return std::nullopt;
	}
	if (inst->GetOpcode() == ValueOpcode::BitwiseOr32) {
		const auto lhs = BallotPredicate(Arg(*inst, 0));
		const auto rhs = BallotPredicate(Arg(*inst, 1));
		return lhs && rhs && *lhs == *rhs ? lhs : std::nullopt;
	}
	if (inst->GetOpcode() != ValueOpcode::CompositeExtractU32x4) {
		return std::nullopt;
	}
	const auto* ballot = InstructionOf(Arg(*inst, 0));
	const auto  word   = Arg(*inst, 1);
	if (ballot == nullptr || ballot->GetOpcode() != ValueOpcode::Ballot || !word.IsImmediate() ||
	    word.GetType() != Type::U32 || word.U32() > 1u) {
		return std::nullopt;
	}
	return Arg(*ballot, 0);
}

// An EXECZ/EXECNZ branch condition: a test of Ballot(predicate) against zero, possibly negated
// and passed through a U32 flag (INotEqual32(SelectU32(test, 1, 0), 0)).
struct ExecTest {
	Value predicate;
	bool  true_when_empty = true;
};

std::optional<ExecTest> MatchExecTest(Value condition, uint32_t depth = 0) {
	const auto* inst = InstructionOf(condition);
	if (inst == nullptr || depth > 8u) {
		return std::nullopt;
	}
	const auto flip = [](std::optional<ExecTest> test) {
		if (test) {
			test->true_when_empty = !test->true_when_empty;
		}
		return test;
	};
	const auto opcode = inst->GetOpcode();
	if (opcode == ValueOpcode::LogicalNot) {
		return flip(MatchExecTest(Arg(*inst, 0), depth + 1u));
	}
	if (opcode != ValueOpcode::IEqual32 && opcode != ValueOpcode::INotEqual32) {
		return std::nullopt;
	}
	auto lhs = Arg(*inst, 0);
	auto rhs = Arg(*inst, 1);
	if (lhs.IsImmediate()) {
		std::swap(lhs, rhs);
	}
	if (!IsU32(rhs, 0u)) {
		return std::nullopt;
	}
	const bool  equal = opcode == ValueOpcode::IEqual32;
	const auto* flag  = InstructionOf(lhs);
	if (flag != nullptr && flag->GetOpcode() == ValueOpcode::SelectU32 &&
	    IsU32(Arg(*flag, 1), 1u) && IsU32(Arg(*flag, 2), 0u)) {
		const auto test = MatchExecTest(Arg(*flag, 0), depth + 1u);
		return equal ? flip(test) : test;
	}
	const auto predicate = BallotPredicate(lhs);
	if (!predicate) {
		return std::nullopt;
	}
	return ExecTest {.predicate = *predicate, .true_when_empty = equal};
}

// Whether `condition` can only hold where `predicate` holds: the predicate itself, false, or an
// AND with an operand that implies it (EXEC narrowed inside the region).
bool Implies(Value condition, Value predicate, uint32_t depth = 0) {
	condition = condition.Resolve();
	if (condition == predicate) {
		return true;
	}
	if (condition.IsImmediate()) {
		return condition.GetType() == Type::U1 && !condition.U1();
	}
	const auto* inst = condition.TryInstruction();
	if (inst == nullptr || depth > 32u || inst->GetOpcode() != ValueOpcode::LogicalAnd) {
		return false;
	}
	return Implies(Arg(*inst, 0), predicate, depth + 1u) ||
	       Implies(Arg(*inst, 1), predicate, depth + 1u);
}

// Whether `value` equals `skipped` in every lane where `predicate` is false: EXEC-masked writes
// select(q, new, old) with q implying the predicate, applied over `skipped`, possibly through phis
// inside the region (loops), where a cycle back to a phi being checked holds by induction.
bool SameWhereOff(Value value, Value skipped, Value predicate,
                  const std::unordered_set<const Block*>& region,
                  std::vector<const Inst*>& visiting) {
	value = value.Resolve();
	if (value == skipped) {
		return true;
	}
	const auto* inst = value.TryInstruction();
	if (inst == nullptr || visiting.size() > 256u) {
		return false;
	}
	switch (inst->GetOpcode()) {
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectF32:
			return Implies(Arg(*inst, 0), predicate) &&
			       SameWhereOff(Arg(*inst, 2), skipped, predicate, region, visiting);
		case ValueOpcode::Phi: {
			if (!region.contains(inst->Parent())) {
				return false;
			}
			if (std::ranges::find(visiting, inst) != visiting.end()) {
				return true;
			}
			visiting.push_back(inst);
			bool same = true;
			for (size_t index = 0; same && index < inst->NumArgs(); index++) {
				same = SameWhereOff(Arg(*inst, index), skipped, predicate, region, visiting);
			}
			visiting.pop_back();
			return same;
		}
		default: return false;
	}
}

// Values that can differ between lanes. Everything else is the same in every lane: constants,
// user data, SRT and constant-buffer reads, ballots and lane reads, and pure arithmetic or phis
// over those. Before this pass every branch in the IR is uniform, so a phi of uniform values is
// uniform; a converted region leaves every value the same per lane as before.
bool AlwaysUniform(ValueOpcode opcode) {
	switch (opcode) {
		case ValueOpcode::Ballot:
		case ValueOpcode::ReadFirstLane:
		case ValueOpcode::ReadLane: return true;
		default: return false;
	}
}

bool UniformFromOperands(ValueOpcode opcode) {
	switch (opcode) {
		case ValueOpcode::Phi:
		case ValueOpcode::Identity:
		case ValueOpcode::GetUserData:
		case ValueOpcode::GetShaderBase:
		case ValueOpcode::ReadConst:
		case ValueOpcode::ReadConstBuffer:
		case ValueOpcode::GetSrtResource:
		case ValueOpcode::GetScratchResource:
		case ValueOpcode::GetBufferResource:
		case ValueOpcode::GetAddressResource:
		case ValueOpcode::GetImageResource:
		case ValueOpcode::GetSamplerResource: return true;
		default: return IsLaneLocalOpcode(opcode);
	}
}

std::unordered_set<const Inst*> VaryingValues(const Program& program) {
	std::unordered_set<const Inst*> varying;
	std::vector<const Inst*>        pending;
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			const auto opcode = inst.GetOpcode();
			if (!AlwaysUniform(opcode) && !UniformFromOperands(opcode)) {
				varying.insert(&inst);
				pending.push_back(&inst);
			}
		}
	}
	while (!pending.empty()) {
		const auto* inst = pending.back();
		pending.pop_back();
		for (const auto& use: inst->Uses()) {
			if (UniformFromOperands(use.user->GetOpcode()) && varying.insert(use.user).second) {
				pending.push_back(use.user);
			}
		}
	}
	return varying;
}

// Whether an instruction still means the same when only the lanes where `predicate` holds run
// it. Anything that reads other lanes needs them all: implicit-LOD samples and derivatives need
// whole quads, and ReadLane, DPP, swizzles and appends address lanes the region may not run.
// Stores and atomics must already be masked by the predicate. Exports, attribute reads and
// barriers stay outside regions.
bool AllowedInRegion(const Program& program, const Inst& inst, Value predicate) {
	const auto opcode = inst.GetOpcode();
	if (IsLaneLocalOpcode(opcode)) {
		return true;
	}
	switch (opcode) {
		case ValueOpcode::Phi:
		case ValueOpcode::Identity:
		case ValueOpcode::Void:
		case ValueOpcode::Reference:
		case ValueOpcode::ReferenceU32:
		case ValueOpcode::ControlNop:
		case ValueOpcode::Waitcnt:
		case ValueOpcode::TtraceData:
		case ValueOpcode::InstPrefetch:
		case ValueOpcode::UndefU1:
		case ValueOpcode::UndefU8:
		case ValueOpcode::UndefU16:
		case ValueOpcode::UndefU32:
		case ValueOpcode::UndefU64:
		case ValueOpcode::LaneId:
		case ValueOpcode::WriteLane: // A select on the lane's own id.
		case ValueOpcode::GetUserData:
		case ValueOpcode::GetShaderBase:
		case ValueOpcode::GetBuiltin:
		case ValueOpcode::ReadConst:
		case ValueOpcode::ReadConstBuffer:
		case ValueOpcode::GetSrtResource:
		case ValueOpcode::GetBufferResource:
		case ValueOpcode::GetAddressResource:
		case ValueOpcode::GetScratchResource:
		case ValueOpcode::GetImageResource:
		case ValueOpcode::GetSamplerResource:
		case ValueOpcode::MakeImageAddress:
		case ValueOpcode::ImageQueryDimensions:
		case ValueOpcode::ImageRead:
		case ValueOpcode::ImageGatherRaw: return true; // Gathers read the base level.
		case ValueOpcode::ImageSampleRaw: {
			const auto index = inst.Flags<MemoryFlags>().index;
			if (index >= program.memory_info.size()) {
				return false;
			}
			constexpr uint32_t explicit_lod = Decoder::ImageSampleFlagDerivative |
			                                  Decoder::ImageSampleFlagLod |
			                                  Decoder::ImageSampleFlagLevelZero;
			return (program.memory_info[index].image_sample_flags & explicit_lod) != 0u;
		}
		case ValueOpcode::Ballot: return Implies(Arg(inst, 0), predicate);
		case ValueOpcode::ReadFirstLane: return Implies(Arg(inst, 1), predicate);
		default: break;
	}
	const auto buffer  = BufferAccessOf(opcode);
	const auto shared  = SharedAccessOf(opcode);
	const auto address = AddressOpcodeInfoOf(opcode).access;
	const auto image   = ImageOpcodeInfoOf(opcode).access;
	if (buffer == BufferAccess::Read || shared == SharedAccess::Read ||
	    address == AddressAccess::Read) {
		return true;
	}
	const bool masked_write = buffer == BufferAccess::Write || buffer == BufferAccess::Atomic ||
	                          shared == SharedAccess::Write || shared == SharedAccess::Atomic ||
	                          address == AddressAccess::Write || image == ImageAccess::Write ||
	                          image == ImageAccess::Atomic;
	return masked_write && inst.NumArgs() != 0u &&
	       Implies(Arg(inst, inst.NumArgs() - 1u), predicate);
}

class Brancher {
public:
	explicit Brancher(Program& program): m_program(program), m_varying(VaryingValues(program)) {
		for (uint32_t index = 0; index < program.blocks.size(); index++) {
			m_index_of_id.emplace(program.block_info[index].id, index);
			m_index_of_block.emplace(program.blocks[index], index);
		}
	}

	bool TryConvert(uint32_t header) {
		auto&       info = m_program.block_info[header];
		const auto& term = info.terminator;
		if (term.kind != CFG::TerminatorKind::ConditionalBranch || term.loop_header ||
		    term.merge_block == UINT32_MAX || info.condition.IsEmpty() ||
		    (term.condition != CFG::BranchCondition::ExecZero &&
		     term.condition != CFG::BranchCondition::ExecNonZero)) {
			return false;
		}
		const auto test = MatchExecTest(info.condition);
		if (!test || test->predicate.IsImmediate()) {
			return false;
		}
		// The skip edge goes straight to the merge; the other edge enters the region.
		const auto skip_id = test->true_when_empty ? term.true_block : term.false_block;
		const auto body_id = test->true_when_empty ? term.false_block : term.true_block;
		const auto merge   = IndexOf(term.merge_block);
		const auto body    = IndexOf(body_id);
		if (skip_id != term.merge_block || merge == UINT32_MAX || body == UINT32_MAX ||
		    body == merge) {
			return false;
		}
		std::vector<uint32_t>            blocks;
		std::unordered_set<const Block*> region;
		if (!CollectRegion(header, body, merge, blocks, region)) {
			return false;
		}
		const auto predicate = test->predicate;
		for (const auto block: blocks) {
			for (const auto& inst: *m_program.blocks[block]) {
				if (!AllowedInRegion(m_program, inst, predicate)) {
					return false;
				}
			}
		}
		if (!StaysInRegion(blocks, region, merge)) {
			return false;
		}
		std::vector<Inst*> uniform_results;
		if (!CheckEscaping(header, merge, predicate, region, uniform_results)) {
			return false;
		}
		Convert(header, merge, *test, blocks, uniform_results);
		return true;
	}

private:
	uint32_t IndexOf(uint32_t id) const {
		const auto found = m_index_of_id.find(id);
		return found != m_index_of_id.end() ? found->second : UINT32_MAX;
	}

	uint32_t IndexOf(const Block* block) const {
		const auto found = m_index_of_block.find(block);
		return found != m_index_of_block.end() ? found->second : UINT32_MAX;
	}

	// The blocks reachable from the body without passing the merge. Each must branch (a return
	// would end lanes that now skip the region instead) and every edge out of them must end at
	// the merge; every edge into the merge must come from them or the header.
	bool CollectRegion(uint32_t header, uint32_t body, uint32_t merge,
	                   std::vector<uint32_t>& blocks, std::unordered_set<const Block*>& region) {
		std::vector<uint32_t> pending {body};
		while (!pending.empty()) {
			const auto block = pending.back();
			pending.pop_back();
			if (block == header || block == UINT32_MAX) {
				return false;
			}
			if (block == merge || region.contains(m_program.blocks[block])) {
				continue;
			}
			const auto& info       = m_program.block_info[block];
			const auto  successors = m_program.blocks[block]->ImmSuccessors();
			const bool  branches =
			    info.terminator.kind == CFG::TerminatorKind::Branch ||
			    (info.terminator.kind == CFG::TerminatorKind::ConditionalBranch &&
			     !info.condition.IsEmpty());
			if (!branches || successors.empty()) {
				return false;
			}
			region.insert(m_program.blocks[block]);
			blocks.push_back(block);
			for (const auto* successor: successors) {
				pending.push_back(IndexOf(successor));
			}
		}
		for (const auto* predecessor: m_program.blocks[merge]->ImmPredecessors()) {
			if (predecessor != m_program.blocks[header] && !region.contains(predecessor)) {
				return false;
			}
		}
		return !blocks.empty();
	}

	// Region values may only be used inside the region or by the merge's phis.
	bool StaysInRegion(const std::vector<uint32_t>&            blocks,
	                   const std::unordered_set<const Block*>& region, uint32_t merge) const {
		std::unordered_set<const Inst*> defined;
		for (const auto block: blocks) {
			for (const auto& inst: *m_program.blocks[block]) {
				defined.insert(&inst);
				for (const auto& use: inst.Uses()) {
					const auto* parent = use.user->Parent();
					if (!region.contains(parent) &&
					    (parent != m_program.blocks[merge] ||
					     use.user->GetOpcode() != ValueOpcode::Phi)) {
						return false;
					}
				}
			}
		}
		for (uint32_t index = 0; index < m_program.block_info.size(); index++) {
			if (region.contains(m_program.blocks[index])) {
				continue;
			}
			const auto& info = m_program.block_info[index];
			for (const auto& value: {info.condition, info.indirect_target}) {
				if (!value.IsEmpty() && defined.contains(InstructionOf(value))) {
					return false;
				}
			}
		}
		return true;
	}

	// Values leave the region only through the merge's phis. On a lane that skips the region such
	// a phi now takes the header's value, where before it took what the region computed: they
	// must be the same (EXEC-masked writes), or the value must be uniform, which is then read
	// from a lane that ran the region.
	bool CheckEscaping(uint32_t header, uint32_t merge, Value predicate,
	                   const std::unordered_set<const Block*>& region,
	                   std::vector<Inst*>&                     uniform_results) const {
		std::vector<const Inst*> visiting;
		for (auto& inst: *m_program.blocks[merge]) {
			if (inst.GetOpcode() != ValueOpcode::Phi) {
				continue;
			}
			std::optional<Value> skipped;
			for (size_t index = 0; index < inst.NumPhiBlocks(); index++) {
				if (inst.PhiBlock(index) == m_program.blocks[header]) {
					skipped = Arg(inst, index);
				}
			}
			if (!skipped) {
				return false;
			}
			bool uniform = false;
			for (size_t index = 0; index < inst.NumPhiBlocks(); index++) {
				if (!region.contains(inst.PhiBlock(index))) {
					continue;
				}
				const auto value = Arg(inst, index);
				visiting.clear();
				if (SameWhereOff(value, *skipped, predicate, region, visiting)) {
					continue;
				}
				const auto type = inst.GetType();
				if (!IsUniform(value) || (type != Type::U32 && type != Type::U1)) {
					return false;
				}
				uniform = true;
			}
			if (uniform) {
				uniform_results.push_back(&inst);
			}
		}
		return true;
	}

	bool IsUniform(Value value) const {
		value = value.Resolve();
		if (value.IsImmediate()) {
			return true;
		}
		const auto* inst = value.TryInstruction();
		return inst != nullptr && !m_varying.contains(inst);
	}

	void Convert(uint32_t header, uint32_t merge, const ExecTest& test,
	             const std::vector<uint32_t>& blocks, const std::vector<Inst*>& uniform_results) {
		auto&       info          = m_program.block_info[header];
		auto*       header_block  = m_program.blocks[header];
		const auto  predicate     = test.predicate;
		const Value old_condition = info.condition.Resolve();

		// A lane's own EXEC bit now decides whether it runs the region. Ballots leave out helper
		// lanes, so a helper lane with the bit now runs the region even when no other lane has it,
		// as on hardware; helper lanes only feed derivatives.
		const Value lane_condition =
		    test.true_when_empty
		        ? Value(&header_block->AppendNewInst(ValueOpcode::LogicalNot, {predicate}))
		        : predicate;
		header_block->AppendNewInst(ValueOpcode::Reference, {lane_condition});
		info.condition = lane_condition;
		if (const auto* inst = lane_condition.TryInstruction(); inst != nullptr) {
			m_varying.insert(inst);
		}

		// Uniform results: read them from the first lane that ran the region, unless none did.
		if (!uniform_results.empty()) {
			const Value empty_mask =
			    test.true_when_empty
			        ? old_condition
			        : Value(&header_block->AppendNewInst(ValueOpcode::LogicalNot, {old_condition}));
			auto* merge_block = m_program.blocks[merge];
			const auto not_phi = [](const Inst& inst) { return inst.GetOpcode() != ValueOpcode::Phi; };
			auto       insert  = std::ranges::find_if(*merge_block, not_phi);
			for (auto* phi: uniform_results) {
				const auto  uses = phi->Uses();
				const Value phi_value(phi);
				Value       fixed;
				Inst*       word = nullptr;
				if (phi->GetType() == Type::U32) {
					const Value read(&*merge_block->PrependNewInst(
					    insert, ValueOpcode::ReadFirstLane, {phi_value, predicate}));
					fixed = Value(&*merge_block->PrependNewInst(insert, ValueOpcode::SelectU32,
					                                           {empty_mask, phi_value, read}));
				} else {
					word = &*merge_block->PrependNewInst(insert, ValueOpcode::SelectU32,
					                                     {phi_value, Value(1u), Value(0u)});
					const Value read(&*merge_block->PrependNewInst(
					    insert, ValueOpcode::ReadFirstLane, {Value(word), predicate}));
					const Value bit(&*merge_block->PrependNewInst(
					    insert, ValueOpcode::INotEqual32, {read, Value(0u)}));
					fixed = Value(&*merge_block->PrependNewInst(insert, ValueOpcode::SelectU1,
					                                           {empty_mask, phi_value, bit}));
				}
				for (const auto& use: uses) {
					use.user->SetArg(use.operand, fixed);
				}
				// For later regions: the phi now differs between lanes that ran the region and
				// lanes that skipped it, and the fixed value is uniform exactly when the phi was.
				if (m_varying.contains(phi)) {
					m_varying.insert(fixed.TryInstruction());
				}
				m_varying.insert(phi);
				if (word != nullptr) {
					m_varying.insert(word);
				}
				for (auto& block_info: m_program.block_info) {
					if (!block_info.condition.IsEmpty() &&
					    block_info.condition.Resolve() == phi_value) {
						block_info.condition = fixed;
						merge_block->PrependNewInst(insert, ValueOpcode::Reference, {fixed});
					}
					if (!block_info.indirect_target.IsEmpty() &&
					    block_info.indirect_target.Resolve() == phi_value) {
						block_info.indirect_target = fixed;
						merge_block->PrependNewInst(insert, ValueOpcode::ReferenceU32, {fixed});
					}
				}
			}
		}

		// Every lane inside the region has the predicate.
		for (const auto block: blocks) {
			for (auto& inst: *m_program.blocks[block]) {
				for (size_t index = 0; index < inst.NumArgs(); index++) {
					if (inst.Arg(index).Resolve() == predicate) {
						inst.SetArg(index, Value(true));
					}
				}
			}
		}
	}

	Program&                                   m_program;
	std::unordered_set<const Inst*>            m_varying;
	std::unordered_map<uint32_t, uint32_t>     m_index_of_id;
	std::unordered_map<const Block*, uint32_t> m_index_of_block;
};

} // namespace

uint32_t BranchExecRegions(Program& program) {
	// Pixel shaders only: compute and mesh shaders may run a wave64 guest as two host halves,
	// where a branch condition is rebuilt from both halves' ballots, and the dispatcher fallback
	// has no structured merges.
	if (program.stage != ShaderType::Pixel || program.dispatcher_fallback ||
	    program.block_info.size() != program.blocks.size()) {
		return 0;
	}
	Brancher brancher(program);
	uint32_t converted = 0;
	for (uint32_t header = 0; header < program.blocks.size(); header++) {
		converted += brancher.TryConvert(header) ? 1u : 0u;
	}
	if (converted != 0u) {
		ConstantPropagationPass(program.blocks);
		ResolveControlFlowIdentities(program);
		RemoveIdentities(program.blocks);
		EliminateDeadCode(program.blocks);
	}
	return converted;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
