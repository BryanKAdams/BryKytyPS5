#include "graphics/shader/recompiler/ir/passes/ConstantPropagation.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

Value Arg(const Inst& inst, size_t index) {
	return inst.Arg(index).Resolve();
}

bool IsImmediate(Value value, Type type) {
	return value.IsImmediate() && value.GetType() == type;
}

void Replace(Inst& inst, Value value) {
	inst.ReplaceUsesWith(value.Resolve());
}

template <typename Function>
bool FoldU32(Inst& inst, Function function) {
	const auto lhs = Arg(inst, 0);
	const auto rhs = Arg(inst, 1);
	if (!IsImmediate(lhs, Type::U32) || !IsImmediate(rhs, Type::U32)) {
		return false;
	}
	Replace(inst, Value(static_cast<uint32_t>(function(lhs.U32(), rhs.U32()))));
	return true;
}

template <typename Function>
bool FoldU64(Inst& inst, Function function) {
	const auto lhs = Arg(inst, 0);
	const auto rhs = Arg(inst, 1);
	if (!IsImmediate(lhs, Type::U64) || !IsImmediate(rhs, Type::U64)) {
		return false;
	}
	Replace(inst, Value(static_cast<uint64_t>(function(lhs.U64(), rhs.U64()))));
	return true;
}

template <typename Function>
bool FoldU64Shift(Inst& inst, Function function) {
	const auto value = Arg(inst, 0);
	const auto shift = Arg(inst, 1);
	if (!IsImmediate(value, Type::U64) || !IsImmediate(shift, Type::U32)) {
		return false;
	}
	Replace(inst, Value(static_cast<uint64_t>(function(value.U64(), shift.U32()))));
	return true;
}

template <typename Function>
bool FoldU32Compare(Inst& inst, Function function) {
	const auto lhs = Arg(inst, 0);
	const auto rhs = Arg(inst, 1);
	if (!IsImmediate(lhs, Type::U32) || !IsImmediate(rhs, Type::U32)) {
		return false;
	}
	Replace(inst, Value(function(lhs.U32(), rhs.U32())));
	return true;
}

template <typename Function>
bool FoldU64Compare(Inst& inst, Function function) {
	const auto lhs = Arg(inst, 0);
	const auto rhs = Arg(inst, 1);
	if (!IsImmediate(lhs, Type::U64) || !IsImmediate(rhs, Type::U64)) {
		return false;
	}
	Replace(inst, Value(function(lhs.U64(), rhs.U64())));
	return true;
}

template <typename Function>
bool FoldLogical(Inst& inst, Function function) {
	const auto lhs = Arg(inst, 0);
	const auto rhs = Arg(inst, 1);
	if (!IsImmediate(lhs, Type::U1) || !IsImmediate(rhs, Type::U1)) {
		return false;
	}
	Replace(inst, Value(function(lhs.U1(), rhs.U1())));
	return true;
}

bool ReplaceBinaryIdentity(Inst& inst, Type type, uint64_t identity) {
	const auto lhs = Arg(inst, 0);
	const auto rhs = Arg(inst, 1);
	if (IsImmediate(lhs, type) &&
	    (type == Type::U32 ? lhs.U32() == identity : lhs.U64() == identity)) {
		Replace(inst, rhs);
		return true;
	}
	if (IsImmediate(rhs, type) &&
	    (type == Type::U32 ? rhs.U32() == identity : rhs.U64() == identity)) {
		Replace(inst, lhs);
		return true;
	}
	return false;
}

bool FoldSelect(Inst& inst) {
	const auto condition   = Arg(inst, 0);
	const auto true_value  = Arg(inst, 1);
	const auto false_value = Arg(inst, 2);
	if (IsImmediate(condition, Type::U1)) {
		Replace(inst, condition.U1() ? true_value : false_value);
		return true;
	}
	if (true_value == false_value) {
		Replace(inst, true_value);
		return true;
	}
	return false;
}

bool FoldPhi(Inst& inst) {
	Value same;
	for (size_t index = 0; index < inst.NumArgs(); index++) {
		const auto value = Arg(inst, index);
		if (value.TryInstruction() == &inst) {
			continue;
		}
		if (same.IsEmpty()) {
			same = value;
		} else if (same != value) {
			return false;
		}
	}
	if (same.IsEmpty()) {
		return false;
	}
	Replace(inst, same);
	return true;
}

bool FoldBitCast(Inst& inst, ValueOpcode reverse) {
	const auto value = Arg(inst, 0);
	if (IsImmediate(value, Type::F32) && inst.GetOpcode() == ValueOpcode::BitCastU32F32) {
		Replace(inst, Value(std::bit_cast<uint32_t>(value.F32Value())));
		return true;
	}
	if (IsImmediate(value, Type::U32) && inst.GetOpcode() == ValueOpcode::BitCastF32U32) {
		Replace(inst, Value::F32(std::bit_cast<float>(value.U32())));
		return true;
	}
	if (IsImmediate(value, Type::F16) && inst.GetOpcode() == ValueOpcode::BitCastU16F16) {
		Replace(inst, Value(value.F16Bits()));
		return true;
	}
	if (IsImmediate(value, Type::U16) && inst.GetOpcode() == ValueOpcode::BitCastF16U16) {
		Replace(inst, Value::F16(value.U16()));
		return true;
	}
	if (auto* producer = value.TryInstruction();
	    producer != nullptr && producer->GetOpcode() == reverse) {
		Replace(inst, producer->Arg(0));
		return true;
	}
	return false;
}

bool FoldCompositeExtract(Inst& inst, ValueOpcode construct, size_t components) {
	const auto composite = Arg(inst, 0);
	const auto index     = Arg(inst, 1);
	if (!IsImmediate(index, Type::U32) || index.U32() >= components) {
		return false;
	}
	const auto  component = index.U32();
	const auto* producer  = composite.TryInstruction();
	if (IsImmediate(composite, Type::U64)) {
		Replace(inst, Value(static_cast<uint32_t>(composite.U64() >> (component * 32u))));
		return true;
	}
	if (producer != nullptr && producer->GetOpcode() == construct) {
		Replace(inst, producer->Arg(component));
		return true;
	}
	if (component < 2u && producer != nullptr &&
	    producer->GetOpcode() == ValueOpcode::IAddCarry32) {
		const auto lhs = producer->Arg(0).Resolve();
		const auto rhs = producer->Arg(1).Resolve();
		if (IsImmediate(lhs, Type::U32) && IsImmediate(rhs, Type::U32)) {
			const auto sum = static_cast<uint64_t>(lhs.U32()) + rhs.U32();
			Replace(inst, Value(component == 0u ? static_cast<uint32_t>(sum)
			                                    : static_cast<uint32_t>(sum >> 32u)));
			return true;
		}
	}
	return false;
}

// Every value a U32 can take, when that set is small: small bit fields and masks, arithmetic on
// such sets, selects and phis. V_MOVRELS/V_MOVRELD compare M0 against every register index, but
// M0 is usually a small field times a stride, so most of those compares are always false.
class PossibleValues {
public:
	static constexpr size_t MaxValues = 64;
	static constexpr size_t MaxFieldBits = 6;

	// Sorted and unique; false when unknown or too large.
	bool Find(Value value, std::vector<uint32_t>& values, uint32_t depth = 0) {
		value = value.Resolve();
		if (IsImmediate(value, Type::U32)) {
			values.assign(1, value.U32());
			return true;
		}
		const auto* inst = value.TryInstruction();
		if (inst == nullptr || depth > 24 || inst->GetType() != Type::U32) {
			return false;
		}
		if (const auto found = m_known.find(inst); found != m_known.end()) {
			values = found->second;
			return !values.empty();
		}
		if (std::ranges::find(m_visiting, inst) != m_visiting.end()) {
			return false; // A loop-carried value can take any value the loop produces.
		}
		m_visiting.push_back(inst);
		const bool known = Compute(*inst, values, depth + 1);
		m_visiting.pop_back();
		if (!known) {
			values.clear();
		}
		m_known.emplace(inst, values);
		return known;
	}

private:
	template <typename Function>
	bool Pairwise(const Inst& inst, std::vector<uint32_t>& values, uint32_t depth,
	              Function function) {
		std::vector<uint32_t> lhs;
		std::vector<uint32_t> rhs;
		if (!Find(inst.Arg(0), lhs, depth) || !Find(inst.Arg(1), rhs, depth)) {
			return false;
		}
		values.clear();
		for (const auto a: lhs) {
			for (const auto b: rhs) {
				values.push_back(function(a, b));
			}
		}
		return Normalize(values);
	}

	static bool Normalize(std::vector<uint32_t>& values) {
		std::ranges::sort(values);
		values.erase(std::unique(values.begin(), values.end()), values.end());
		return !values.empty() && values.size() <= MaxValues;
	}

	bool Compute(const Inst& inst, std::vector<uint32_t>& values, uint32_t depth) {
		switch (inst.GetOpcode()) {
			case ValueOpcode::BitFieldUExtract: {
				std::vector<uint32_t> offset;
				std::vector<uint32_t> count;
				if (!Find(inst.Arg(1), offset, depth) || !Find(inst.Arg(2), count, depth) ||
				    offset.size() != 1u || count.size() != 1u || offset[0] > 32u ||
				    count[0] > 32u - offset[0]) {
					return false;
				}
				const auto bits = count[0];
				if (std::vector<uint32_t> source; Find(inst.Arg(0), source, depth)) {
					for (auto& value: source) {
						value = bits == 0u ? 0u
						                   : (value >> offset[0]) &
						                         (bits == 32u ? UINT32_MAX : (1u << bits) - 1u);
					}
					values = std::move(source);
					return Normalize(values);
				}
				if (bits > MaxFieldBits) {
					return false;
				}
				values.resize(size_t {1} << bits);
				std::iota(values.begin(), values.end(), 0u);
				return true;
			}
			case ValueOpcode::BitwiseAnd32: {
				if (Pairwise(inst, values, depth, [](uint32_t a, uint32_t b) { return a & b; })) {
					return true;
				}
				// A small mask bounds the result even when the other operand is unknown.
				for (uint32_t side = 0; side < 2u; side++) {
					std::vector<uint32_t> mask;
					if (!Find(inst.Arg(side), mask, depth) || mask.size() != 1u ||
					    static_cast<size_t>(std::popcount(mask[0])) > MaxFieldBits) {
						continue;
					}
					// Every submask of the mask.
					values.clear();
					uint32_t subset = mask[0];
					do {
						values.push_back(subset);
						subset = (subset - 1u) & mask[0];
					} while (subset != mask[0]);
					return Normalize(values);
				}
				return false;
			}
			case ValueOpcode::BitwiseOr32:
				return Pairwise(inst, values, depth, [](uint32_t a, uint32_t b) { return a | b; });
			case ValueOpcode::BitwiseXor32:
				return Pairwise(inst, values, depth, [](uint32_t a, uint32_t b) { return a ^ b; });
			case ValueOpcode::IAdd32:
				return Pairwise(inst, values, depth, [](uint32_t a, uint32_t b) { return a + b; });
			case ValueOpcode::ISub32:
				return Pairwise(inst, values, depth, [](uint32_t a, uint32_t b) { return a - b; });
			case ValueOpcode::IMul32:
				return Pairwise(inst, values, depth, [](uint32_t a, uint32_t b) { return a * b; });
			case ValueOpcode::ShiftLeftLogical32:
				return Pairwise(inst, values, depth,
				                [](uint32_t a, uint32_t b) { return a << (b & 31u); });
			case ValueOpcode::ShiftRightLogical32:
				return Pairwise(inst, values, depth,
				                [](uint32_t a, uint32_t b) { return a >> (b & 31u); });
			case ValueOpcode::UMin32:
				return Pairwise(inst, values, depth,
				                [](uint32_t a, uint32_t b) { return std::min(a, b); });
			case ValueOpcode::UMax32:
				return Pairwise(inst, values, depth,
				                [](uint32_t a, uint32_t b) { return std::max(a, b); });
			case ValueOpcode::SelectU32:
			case ValueOpcode::Phi: {
				values.clear();
				for (size_t index = inst.GetOpcode() == ValueOpcode::Phi ? 0u : 1u;
				     index < inst.NumArgs(); index++) {
					std::vector<uint32_t> operand;
					if (!Find(inst.Arg(index), operand, depth)) {
						return false;
					}
					values.insert(values.end(), operand.begin(), operand.end());
				}
				return Normalize(values);
			}
			default: return false;
		}
	}

	std::unordered_map<const Inst*, std::vector<uint32_t>> m_known;
	std::vector<const Inst*>                               m_visiting;
};

// Folds x == C (or x != C) when C is not among x's possible values, or is its only one.
bool FoldImpossibleEquality(Inst& inst, bool equal, PossibleValues& possible) {
	auto lhs = Arg(inst, 0);
	auto rhs = Arg(inst, 1);
	if (IsImmediate(lhs, Type::U32)) {
		std::swap(lhs, rhs);
	}
	if (!IsImmediate(rhs, Type::U32) || IsImmediate(lhs, Type::U32)) {
		return false;
	}
	std::vector<uint32_t> values;
	if (!possible.Find(lhs, values)) {
		return false;
	}
	if (!std::ranges::binary_search(values, rhs.U32())) {
		Replace(inst, Value(!equal));
		return true;
	}
	if (values.size() == 1u) {
		Replace(inst, Value(equal));
		return true;
	}
	return false;
}

void FoldInstruction(Block& block, Block::iterator instruction,
                      std::unordered_set<Inst*>& lowered_ancillary, PossibleValues& possible) {
	auto& inst = *instruction;
	switch (inst.GetOpcode()) {
		case ValueOpcode::Phi: FoldPhi(inst); return;
		case ValueOpcode::SelectU1:
			if (!FoldSelect(inst) && IsImmediate(Arg(inst, 2), Type::U1) &&
			    !Arg(inst, 2).U1()) {
				auto result = block.PrependNewInst(instruction, ValueOpcode::LogicalAnd,
				                                   {Arg(inst, 0), Arg(inst, 1)});
				Replace(inst, Value(&*result));
			}
			return;
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectF32: FoldSelect(inst); return;
		case ValueOpcode::BitFieldInsert: {
			const auto base   = Arg(inst, 0);
			const auto insert = Arg(inst, 1);
			const auto offset = Arg(inst, 2);
			const auto count  = Arg(inst, 3);
			if (IsImmediate(base, Type::U32) && IsImmediate(insert, Type::U32) &&
			    IsImmediate(offset, Type::U32) && IsImmediate(count, Type::U32) &&
			    offset.U32() <= 32u && count.U32() <= 32u - offset.U32()) {
				if (count.U32() == 0u) {
					Replace(inst, base);
					return;
				}
				const auto mask = count.U32() == 32u
				                      ? UINT32_MAX
				                      : ((uint32_t {1} << count.U32()) - 1u) << offset.U32();
				Replace(inst,
				        Value((base.U32() & ~mask) | ((insert.U32() << offset.U32()) & mask)));
			}
			return;
		}
		case ValueOpcode::BitFieldUExtract:
		case ValueOpcode::BitFieldSExtract: {
			const auto value  = Arg(inst, 0);
			const auto offset = Arg(inst, 1);
			const auto count  = Arg(inst, 2);
			auto* source = value.TryInstruction();
			if (source != nullptr && source->GetOpcode() == ValueOpcode::ShiftLeftLogical32 &&
			    IsImmediate(offset, Type::U32) && IsImmediate(count, Type::U32)) {
				const auto shift = Arg(*source, 1);
				if (IsImmediate(shift, Type::U32) && shift.U32() < 32u &&
				    offset.U32() <= shift.U32() && count.U32() <= shift.U32() - offset.U32()) {
					Replace(inst, Value(0u));
					return;
				}
			}
			if (source != nullptr && source->GetOpcode() == ValueOpcode::GetBuiltin &&
			    source->Arg(0) == Value(static_cast<uint32_t>(StageInputKind::PackedAncillary)) &&
			    IsImmediate(offset, Type::U32) && IsImmediate(count, Type::U32) && count.U32() != 0u) {
				constexpr struct {
					uint32_t       start;
					uint32_t       end;
					StageInputKind kind;
				} fields[] = {{8u, 12u, StageInputKind::SampleId}, {16u, 27u, StageInputKind::Layer}};
				for (const auto& field: fields) {
					if (offset.U32() >= field.start && offset.U32() < field.end &&
					    count.U32() <= field.end - offset.U32()) {
						// Preserve extraction and sign extension while exposing only the used field.
						const auto input = block.PrependNewInst(
						    instruction, ValueOpcode::GetBuiltin,
						    {Value(static_cast<uint32_t>(field.kind)), Value(0u)});
						inst.SetArg(0, Value(&*input));
						inst.SetArg(1, Value(offset.U32() - field.start));
						lowered_ancillary.insert(source);
						return;
					}
				}
			}
			if (!IsImmediate(value, Type::U32) || !IsImmediate(offset, Type::U32) ||
			    !IsImmediate(count, Type::U32) || offset.U32() > 32u ||
			    count.U32() > 32u - offset.U32()) {
				return;
			}
			if (count.U32() == 0u) {
				Replace(inst, Value(0u));
			} else if (inst.GetOpcode() == ValueOpcode::BitFieldUExtract) {
				const auto mask =
				    count.U32() == 32u ? UINT32_MAX : (uint32_t {1} << count.U32()) - 1u;
				Replace(inst, Value((value.U32() >> offset.U32()) & mask));
			} else {
				const auto left = 32u - offset.U32() - count.U32();
				const auto bits = value.U32() << left;
				Replace(inst, Value(static_cast<uint32_t>(std::bit_cast<int32_t>(bits) >>
				                                          (left + offset.U32()))));
			}
			return;
		}
		case ValueOpcode::BitCastU16F16: FoldBitCast(inst, ValueOpcode::BitCastF16U16); return;
		case ValueOpcode::BitCastF16U16: FoldBitCast(inst, ValueOpcode::BitCastU16F16); return;
		case ValueOpcode::BitCastU32F32: FoldBitCast(inst, ValueOpcode::BitCastF32U32); return;
		case ValueOpcode::BitCastF32U32: FoldBitCast(inst, ValueOpcode::BitCastU32F32); return;
		case ValueOpcode::ConvertU16U32: {
			const auto value = Arg(inst, 0);
			if (IsImmediate(value, Type::U32)) {
				Replace(inst, Value(static_cast<uint16_t>(value.U32())));
			}
			return;
		}
		case ValueOpcode::ConvertU32U16: {
			const auto value = Arg(inst, 0);
			if (IsImmediate(value, Type::U16)) {
				Replace(inst, Value(static_cast<uint32_t>(value.U16())));
			} else if (auto* producer = value.TryInstruction();
			           producer != nullptr && producer->GetOpcode() == ValueOpcode::ConvertU16U32) {
				Replace(inst, producer->Arg(0));
			}
			return;
		}
		case ValueOpcode::ConvertU8U32: {
			const auto value = Arg(inst, 0);
			if (IsImmediate(value, Type::U32)) {
				Replace(inst, Value(static_cast<uint8_t>(value.U32())));
			}
			return;
		}
		case ValueOpcode::ConvertU32U8: {
			const auto value = Arg(inst, 0);
			if (IsImmediate(value, Type::U8)) {
				Replace(inst, Value(static_cast<uint32_t>(value.U8())));
			} else if (auto* producer = value.TryInstruction();
			           producer != nullptr && producer->GetOpcode() == ValueOpcode::ConvertU8U32) {
				Replace(inst, producer->Arg(0));
			}
			return;
		}
		case ValueOpcode::CompositeExtractU64:
			FoldCompositeExtract(inst, ValueOpcode::CompositeConstructU64, 2);
			return;
		case ValueOpcode::CompositeExtractU32x2:
			FoldCompositeExtract(inst, ValueOpcode::CompositeConstructU32x2, 2);
			return;
		case ValueOpcode::CompositeExtractU32x3:
			FoldCompositeExtract(inst, ValueOpcode::CompositeConstructU32x3, 3);
			return;
		case ValueOpcode::CompositeExtractU32x4:
			FoldCompositeExtract(inst, ValueOpcode::CompositeConstructU32x4, 4);
			return;
		case ValueOpcode::CompositeConstructU64: {
			const auto low  = Arg(inst, 0);
			const auto high = Arg(inst, 1);
			if (IsImmediate(low, Type::U32) && IsImmediate(high, Type::U32)) {
				Replace(inst, Value(static_cast<uint64_t>(low.U32()) |
				                    (static_cast<uint64_t>(high.U32()) << 32u)));
			}
			return;
		}
		case ValueOpcode::IAdd32:
			if (!FoldU32(inst, [](uint32_t a, uint32_t b) { return a + b; })) {
				ReplaceBinaryIdentity(inst, Type::U32, 0u);
			}
			return;
		case ValueOpcode::IAdd64:
			if (!FoldU64(inst, [](uint64_t a, uint64_t b) { return a + b; })) {
				ReplaceBinaryIdentity(inst, Type::U64, 0u);
			}
			return;
		case ValueOpcode::ISub32: {
			if (FoldU32(inst, [](uint32_t a, uint32_t b) { return a - b; })) {
				return;
			}
			const auto rhs = Arg(inst, 1);
			if (IsImmediate(rhs, Type::U32) && rhs.U32() == 0u) {
				Replace(inst, Arg(inst, 0));
			}
			return;
		}
		case ValueOpcode::ISub64: {
			if (FoldU64(inst, [](uint64_t a, uint64_t b) { return a - b; })) {
				return;
			}
			const auto rhs = Arg(inst, 1);
			if (IsImmediate(rhs, Type::U64) && rhs.U64() == 0u) {
				Replace(inst, Arg(inst, 0));
			}
			return;
		}
		case ValueOpcode::IMul32:
			if (!FoldU32(inst, [](uint32_t a, uint32_t b) { return a * b; })) {
				ReplaceBinaryIdentity(inst, Type::U32, 1u);
			}
			return;
		case ValueOpcode::IMul64:
			if (!FoldU64(inst, [](uint64_t a, uint64_t b) { return a * b; })) {
				ReplaceBinaryIdentity(inst, Type::U64, 1u);
			}
			return;
		case ValueOpcode::WqmU64: {
			const auto value = Arg(inst, 0);
			if (IsImmediate(value, Type::U64)) {
				const auto expand = [](uint32_t word) {
					auto quads = word | (word >> 1u);
					quads |= quads >> 2u;
					return (quads & 0x11111111u) * 0x0fu;
				};
				const auto low  = expand(static_cast<uint32_t>(value.U64()));
				const auto high = expand(static_cast<uint32_t>(value.U64() >> 32u));
				Replace(inst,
				        Value(static_cast<uint64_t>(low) | (static_cast<uint64_t>(high) << 32u)));
			}
			return;
		}
		case ValueOpcode::UDiv32: {
			const auto rhs = Arg(inst, 1);
			if (IsImmediate(rhs, Type::U32) && rhs.U32() == 1u) {
				Replace(inst, Arg(inst, 0));
			} else if (IsImmediate(rhs, Type::U32) && rhs.U32() != 0u) {
				FoldU32(inst, [](uint32_t a, uint32_t b) { return a / b; });
			}
			return;
		}
		case ValueOpcode::SMulHi:
			FoldU32(inst, [](uint32_t a, uint32_t b) {
				const auto product = static_cast<int64_t>(std::bit_cast<int32_t>(a)) *
				                     static_cast<int64_t>(std::bit_cast<int32_t>(b));
				return static_cast<uint32_t>(static_cast<uint64_t>(product) >> 32u);
			});
			return;
		case ValueOpcode::UMulHi:
			FoldU32(inst, [](uint32_t a, uint32_t b) {
				return static_cast<uint32_t>((static_cast<uint64_t>(a) * b) >> 32u);
			});
			return;
		case ValueOpcode::IAbs32: {
			const auto value = Arg(inst, 0);
			if (IsImmediate(value, Type::U32)) {
				Replace(inst,
				        Value((value.U32() & 0x80000000u) != 0u ? 0u - value.U32() : value.U32()));
			}
			return;
		}
		case ValueOpcode::ShiftLeftLogical32:
			if (!FoldU32(inst, [](uint32_t a, uint32_t b) { return a << (b & 31u); })) {
				const auto shift = Arg(inst, 1);
				if (IsImmediate(shift, Type::U32) && (shift.U32() & 31u) == 0u) {
					Replace(inst, Arg(inst, 0));
				}
			}
			return;
		case ValueOpcode::ShiftRightLogical32:
			if (!FoldU32(inst, [](uint32_t a, uint32_t b) { return a >> (b & 31u); })) {
				if (IsImmediate(Arg(inst, 0), Type::U32) && Arg(inst, 0).U32() == 0u) {
					Replace(inst, Value(0u));
					return;
				}
				const auto shift = Arg(inst, 1);
				if (IsImmediate(shift, Type::U32) && (shift.U32() & 31u) == 0u) {
					Replace(inst, Arg(inst, 0));
				}
			}
			return;
		case ValueOpcode::ShiftRightArithmetic32:
			FoldU32(inst, [](uint32_t a, uint32_t b) {
				return static_cast<uint32_t>(std::bit_cast<int32_t>(a) >> (b & 31u));
			});
			return;
		case ValueOpcode::ShiftLeftLogical64:
			if (!FoldU64Shift(inst, [](uint64_t a, uint32_t b) { return a << (b & 63u); })) {
				const auto shift = Arg(inst, 1);
				if (IsImmediate(shift, Type::U32) && (shift.U32() & 63u) == 0u) {
					Replace(inst, Arg(inst, 0));
				}
			}
			return;
		case ValueOpcode::ShiftRightLogical64:
			if (!FoldU64Shift(inst, [](uint64_t a, uint32_t b) { return a >> (b & 63u); })) {
				const auto shift = Arg(inst, 1);
				if (IsImmediate(shift, Type::U32) && (shift.U32() & 63u) == 0u) {
					Replace(inst, Arg(inst, 0));
				}
			}
			return;
		case ValueOpcode::ShiftRightArithmetic64:
			FoldU64Shift(inst, [](uint64_t a, uint32_t b) {
				return static_cast<uint64_t>(std::bit_cast<int64_t>(a) >> (b & 63u));
			});
			return;
		case ValueOpcode::BitwiseAnd32:
			if (!FoldU32(inst, [](uint32_t a, uint32_t b) { return a & b; })) {
				ReplaceBinaryIdentity(inst, Type::U32, 0xffffffffu);
			}
			return;
		case ValueOpcode::BitwiseAnd64:
			if (!FoldU64(inst, [](uint64_t a, uint64_t b) { return a & b; })) {
				ReplaceBinaryIdentity(inst, Type::U64, UINT64_MAX);
			}
			return;
		case ValueOpcode::BitwiseOr32:
		case ValueOpcode::BitwiseXor32:
			if (!FoldU32(inst, [opcode = inst.GetOpcode()](uint32_t a, uint32_t b) {
				    return opcode == ValueOpcode::BitwiseOr32 ? a | b : a ^ b;
			    })) {
				ReplaceBinaryIdentity(inst, Type::U32, 0u);
			}
			return;
		case ValueOpcode::BitwiseNot32: {
			const auto value = Arg(inst, 0);
			if (IsImmediate(value, Type::U32)) {
				Replace(inst, Value(~value.U32()));
			} else if (auto* producer = value.TryInstruction();
			           producer != nullptr && producer->GetOpcode() == ValueOpcode::BitwiseNot32) {
				Replace(inst, producer->Arg(0));
			}
			return;
		}
		case ValueOpcode::BitCount32: {
			const auto value = Arg(inst, 0);
			if (IsImmediate(value, Type::U32)) {
				Replace(inst, Value(static_cast<uint32_t>(std::popcount(value.U32()))));
			}
			return;
		}
		case ValueOpcode::BitCount64: {
			const auto value = Arg(inst, 0);
			if (IsImmediate(value, Type::U64)) {
				Replace(inst, Value(static_cast<uint32_t>(std::popcount(value.U64()))));
			}
			return;
		}
		case ValueOpcode::SMin32:
		case ValueOpcode::SMax32:
			FoldU32(inst, [opcode = inst.GetOpcode()](uint32_t a, uint32_t b) {
				const auto lhs = std::bit_cast<int32_t>(a);
				const auto rhs = std::bit_cast<int32_t>(b);
				return opcode == ValueOpcode::SMin32 ? (lhs < rhs ? a : b) : (lhs > rhs ? a : b);
			});
			return;
		case ValueOpcode::UMin32:
			FoldU32(inst, [](uint32_t a, uint32_t b) { return std::min(a, b); });
			return;
		case ValueOpcode::UMax32:
			FoldU32(inst, [](uint32_t a, uint32_t b) { return std::max(a, b); });
			return;
		case ValueOpcode::IEqual32:
			if (!FoldU32Compare(inst, [](uint32_t a, uint32_t b) { return a == b; })) {
				FoldImpossibleEquality(inst, true, possible);
			}
			return;
		case ValueOpcode::INotEqual32:
			if (!FoldU32Compare(inst, [](uint32_t a, uint32_t b) { return a != b; })) {
				FoldImpossibleEquality(inst, false, possible);
			}
			return;
		case ValueOpcode::ULessThan32:
			FoldU32Compare(inst, [](uint32_t a, uint32_t b) { return a < b; });
			return;
		case ValueOpcode::ULessThanEqual32:
			FoldU32Compare(inst, [](uint32_t a, uint32_t b) { return a <= b; });
			return;
		case ValueOpcode::UGreaterThan32:
			FoldU32Compare(inst, [](uint32_t a, uint32_t b) { return a > b; });
			return;
		case ValueOpcode::UGreaterThanEqual32:
			FoldU32Compare(inst, [](uint32_t a, uint32_t b) { return a >= b; });
			return;
		case ValueOpcode::SLessThan32:
		case ValueOpcode::SLessThanEqual32:
		case ValueOpcode::SGreaterThan32:
		case ValueOpcode::SGreaterThanEqual32:
			FoldU32Compare(inst, [opcode = inst.GetOpcode()](uint32_t a, uint32_t b) {
				const auto lhs = std::bit_cast<int32_t>(a);
				const auto rhs = std::bit_cast<int32_t>(b);
				switch (opcode) {
					case ValueOpcode::SLessThan32: return lhs < rhs;
					case ValueOpcode::SLessThanEqual32: return lhs <= rhs;
					case ValueOpcode::SGreaterThan32: return lhs > rhs;
					default: return lhs >= rhs;
				}
			});
			return;
		case ValueOpcode::IEqual64:
			FoldU64Compare(inst, [](uint64_t a, uint64_t b) { return a == b; });
			return;
		case ValueOpcode::INotEqual64:
			FoldU64Compare(inst, [](uint64_t a, uint64_t b) { return a != b; });
			return;
		case ValueOpcode::ULessThan64:
			FoldU64Compare(inst, [](uint64_t a, uint64_t b) { return a < b; });
			return;
		case ValueOpcode::UGreaterThan64:
			FoldU64Compare(inst, [](uint64_t a, uint64_t b) { return a > b; });
			return;
		case ValueOpcode::SLessThan64:
			FoldU64Compare(inst, [](uint64_t a, uint64_t b) {
				return std::bit_cast<int64_t>(a) < std::bit_cast<int64_t>(b);
			});
			return;
		case ValueOpcode::LogicalAnd:
			if (!FoldLogical(inst, [](bool a, bool b) { return a && b; })) {
				const auto lhs = Arg(inst, 0);
				const auto rhs = Arg(inst, 1);
				const auto simplify = [&](Value assumption, Value expression) {
					const auto* disjunction = expression.TryInstruction();
					if (disjunction == nullptr || disjunction->GetOpcode() != ValueOpcode::LogicalOr) {
						return false;
					}
					for (uint32_t i = 0; i < 2u; ++i) {
						const auto* inverse = disjunction->Arg(i).Resolve().TryInstruction();
						if (inverse != nullptr && inverse->GetOpcode() == ValueOpcode::LogicalNot &&
						    inverse->Arg(0).Resolve() == assumption) {
							inst.SetArg(assumption == lhs ? 1u : 0u,
							            disjunction->Arg(i ^ 1u));
							return true;
						}
					}
					return false;
				};
				if (simplify(lhs, rhs) || simplify(rhs, lhs)) return;
				if (IsImmediate(lhs, Type::U1)) {
					Replace(inst, lhs.U1() ? rhs : lhs);
				} else if (IsImmediate(rhs, Type::U1)) {
					Replace(inst, rhs.U1() ? lhs : rhs);
				}
			}
			return;
		case ValueOpcode::LogicalOr:
			if (!FoldLogical(inst, [](bool a, bool b) { return a || b; })) {
				const auto lhs = Arg(inst, 0);
				const auto rhs = Arg(inst, 1);
				if (IsImmediate(lhs, Type::U1)) {
					Replace(inst, lhs.U1() ? lhs : rhs);
				} else if (IsImmediate(rhs, Type::U1)) {
					Replace(inst, rhs.U1() ? rhs : lhs);
				}
			}
			return;
		case ValueOpcode::LogicalXor:
			if (!FoldLogical(inst, [](bool a, bool b) { return a != b; })) {
				const auto lhs = Arg(inst, 0);
				const auto rhs = Arg(inst, 1);
				if (IsImmediate(lhs, Type::U1) && !lhs.U1()) {
					Replace(inst, rhs);
				} else if (IsImmediate(rhs, Type::U1) && !rhs.U1()) {
					Replace(inst, lhs);
				}
			}
			return;
		case ValueOpcode::LogicalNot: {
			const auto value = Arg(inst, 0);
			if (IsImmediate(value, Type::U1)) {
				Replace(inst, Value(!value.U1()));
			} else if (auto* producer = value.TryInstruction();
			           producer != nullptr && producer->GetOpcode() == ValueOpcode::LogicalNot) {
				Replace(inst, producer->Arg(0));
			}
			return;
		}
		default: return;
	}
}

bool IsSelect(ValueOpcode op) {
	return op == ValueOpcode::SelectU1 || op == ValueOpcode::SelectU32 ||
	       op == ValueOpcode::SelectF32;
}

// Pure per-lane operations: no memory, no side effects, no view of other lanes (so no
// derivatives, DPP, ballots or lane reads). Their value in a lane depends only on their operands'
// values in that lane.
bool IsLaneLocal(ValueOpcode op) {
	switch (op) {
		case ValueOpcode::BitCastU16F16:
		case ValueOpcode::BitCastF16U16:
		case ValueOpcode::BitCastU32F32:
		case ValueOpcode::BitCastF32U32:
		case ValueOpcode::ConvertU16U32:
		case ValueOpcode::ConvertU32U16:
		case ValueOpcode::ConvertU8U32:
		case ValueOpcode::ConvertU32U8:
		case ValueOpcode::ConvertF32F16:
		case ValueOpcode::ConvertF16F32:
		case ValueOpcode::ConvertS32F32:
		case ValueOpcode::ConvertU32F32:
		case ValueOpcode::ConvertF32S32:
		case ValueOpcode::ConvertF32U32:
		case ValueOpcode::CompositeConstructU64:
		case ValueOpcode::CompositeConstructU32x2:
		case ValueOpcode::CompositeConstructU32x3:
		case ValueOpcode::CompositeConstructF32x2:
		case ValueOpcode::CompositeConstructU32x4:
		case ValueOpcode::CompositeExtractU64:
		case ValueOpcode::CompositeExtractU32x2:
		case ValueOpcode::CompositeExtractU32x3:
		case ValueOpcode::CompositeExtractU32x4:
		case ValueOpcode::PackHalf2x16:
		case ValueOpcode::PackSnorm2x16:
		case ValueOpcode::PackUnorm2x16:
		case ValueOpcode::PackFloat2x16Rtz:
		case ValueOpcode::FPAbs32:
		case ValueOpcode::FPNeg32:
		case ValueOpcode::FPSaturate32:
		case ValueOpcode::BitFieldInsert:
		case ValueOpcode::BitFieldUExtract:
		case ValueOpcode::BitFieldSExtract:
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectF32:
		case ValueOpcode::IAdd32:
		case ValueOpcode::IAdd64:
		case ValueOpcode::IAddCarry32:
		case ValueOpcode::ISub32:
		case ValueOpcode::ISub64:
		case ValueOpcode::IMul32:
		case ValueOpcode::IMul64:
		case ValueOpcode::UDiv32:
		case ValueOpcode::SMulHi:
		case ValueOpcode::UMulHi:
		case ValueOpcode::IAbs32:
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
		case ValueOpcode::BitReverse32:
		case ValueOpcode::BitCount32:
		case ValueOpcode::BitCount64:
		case ValueOpcode::FindUMsb32:
		case ValueOpcode::FindUMsb64:
		case ValueOpcode::FindILsb32:
		case ValueOpcode::SMin32:
		case ValueOpcode::UMin32:
		case ValueOpcode::SMax32:
		case ValueOpcode::UMax32:
		case ValueOpcode::SMinTri32:
		case ValueOpcode::UMinTri32:
		case ValueOpcode::SMaxTri32:
		case ValueOpcode::UMaxTri32:
		case ValueOpcode::SMedTri32:
		case ValueOpcode::UMedTri32:
		case ValueOpcode::SLessThan32:
		case ValueOpcode::SLessThan64:
		case ValueOpcode::ULessThan32:
		case ValueOpcode::ULessThan64:
		case ValueOpcode::IEqual32:
		case ValueOpcode::IEqual64:
		case ValueOpcode::SLessThanEqual32:
		case ValueOpcode::ULessThanEqual32:
		case ValueOpcode::SGreaterThan32:
		case ValueOpcode::UGreaterThan32:
		case ValueOpcode::UGreaterThan64:
		case ValueOpcode::INotEqual32:
		case ValueOpcode::INotEqual64:
		case ValueOpcode::SGreaterThanEqual32:
		case ValueOpcode::UGreaterThanEqual32:
		case ValueOpcode::LogicalOr:
		case ValueOpcode::LogicalAnd:
		case ValueOpcode::LogicalXor:
		case ValueOpcode::LogicalNot:
		case ValueOpcode::FPOrdEqual32:
		case ValueOpcode::FPUnordEqual32:
		case ValueOpcode::FPOrdNotEqual32:
		case ValueOpcode::FPUnordNotEqual32:
		case ValueOpcode::FPOrdLessThan32:
		case ValueOpcode::FPUnordLessThan32:
		case ValueOpcode::FPOrdGreaterThan32:
		case ValueOpcode::FPUnordGreaterThan32:
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPUnordLessThanEqual32:
		case ValueOpcode::FPOrdGreaterThanEqual32:
		case ValueOpcode::FPUnordGreaterThanEqual32:
		case ValueOpcode::FPIsNan32:
		case ValueOpcode::FPCmpClass32:
		case ValueOpcode::FPAdd32:
		case ValueOpcode::FPSub32:
		case ValueOpcode::FPFma32:
		case ValueOpcode::FPMul32:
		case ValueOpcode::FPMin32:
		case ValueOpcode::FPMax32:
		case ValueOpcode::FPMinTri32:
		case ValueOpcode::FPMaxTri32:
		case ValueOpcode::FPMedTri32:
		case ValueOpcode::FPRecip32:
		case ValueOpcode::FPRecipSqrt32:
		case ValueOpcode::FPSqrt:
		case ValueOpcode::FPSin:
		case ValueOpcode::FPCos:
		case ValueOpcode::FPExp2:
		case ValueOpcode::FPLog2:
		case ValueOpcode::FPLdexp:
		case ValueOpcode::FPRoundEven32:
		case ValueOpcode::FPFloor32:
		case ValueOpcode::FPCeil32:
		case ValueOpcode::FPTrunc32:
		case ValueOpcode::FPFract32: return !HasSideEffects(op);
		default: return false;
	}
}

// Whether a lane-local value only matters in lanes where the predicate holds: every use is the
// true arm of a select on that predicate, or a lane-local operation that itself only matters there.
// Computed on the current graph for each query, within a node budget.
bool ObservedOnlyWhere(const Inst& inst, Value predicate, std::vector<const Inst*>& seen,
                       uint32_t& budget) {
	if (std::ranges::find(seen, &inst) != seen.end()) {
		return true;
	}
	if (budget == 0u) {
		return false;
	}
	budget--;
	seen.push_back(&inst);
	for (const auto& use: inst.Uses()) {
		const auto& user = *use.user;
		const auto  op   = user.GetOpcode();
		if (op == ValueOpcode::Identity) {
			if (!ObservedOnlyWhere(user, predicate, seen, budget)) {
				return false;
			}
			continue;
		}
		if (IsSelect(op) && use.operand == 1u && Arg(user, 0) == predicate) {
			continue;
		}
		if (!IsLaneLocal(op) || !ObservedOnlyWhere(user, predicate, seen, budget)) {
			return false;
		}
	}
	return true;
}

// EXEC-masked writes become select(exec, new, old), so repeated writes in one masked region nest
// selects on the same predicate. For s = select(p, a, b): a false-arm use by another select on p
// only sees b, and a use whose value only matters where p holds only sees a. Rewriting those uses
// leaves the intermediate selects (and their bit casts) dead.
void CollapseSelectChains(const BlockList& blocks) {
	std::vector<Use>         uses;
	std::vector<const Inst*> seen;
	for (auto* block: blocks) {
		for (auto& inst: *block) {
			if (!IsSelect(inst.GetOpcode())) {
				continue;
			}
			const auto predicate = Arg(inst, 0);
			if (predicate.IsImmediate()) {
				continue;
			}
			const auto if_true  = Arg(inst, 1);
			const auto if_false = Arg(inst, 2);
			uses                = inst.Uses();
			for (const auto& use: uses) {
				auto&      user = *use.user;
				const auto op   = user.GetOpcode();
				if (IsSelect(op) && use.operand != 0u && Arg(user, 0) == predicate) {
					user.SetArg(use.operand, use.operand == 1u ? if_true : if_false);
					continue;
				}
				uint32_t budget = 64;
				seen.clear();
				if (IsLaneLocal(op) && ObservedOnlyWhere(user, predicate, seen, budget)) {
					user.SetArg(use.operand, if_true);
				}
			}
		}
	}
}

} // namespace

void ConstantPropagationPass(const BlockList& blocks) {
	std::unordered_set<Inst*> lowered_ancillary;
	PossibleValues            possible;
	for (auto* block: blocks) {
		for (auto inst = block->begin(); inst != block->end(); ++inst) {
			FoldInstruction(*block, inst, lowered_ancillary, possible);
		}
	}
	CollapseSelectChains(blocks);
	// Normalize retained PHI/select values only after every supported field read has
	// been lowered; direct raw consumers remain unsupported.
	for (auto* source: lowered_ancillary) {
		const bool retained_only = std::ranges::all_of(source->Uses(), [](const Use& use) {
			const auto& user = *use.user;
			if (!user.HasUses() && !user.MayHaveSideEffects()) {
				return true;
			}
			return user.GetOpcode() == ValueOpcode::Phi ||
			       (user.GetOpcode() == ValueOpcode::SelectU32 && use.operand == 2u);
		});
		if (retained_only) {
			Replace(*source, Value(0u));
		}
	}
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
