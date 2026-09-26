#include "common/assert.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/frontend/translate/Translator.h"

#include <algorithm>
#include <array>
#include <bit>
#include <utility>

namespace Libs::Graphics::ShaderRecompiler::Frontend {

namespace {

IR::ExportTargetKind ExportTargetKindFromTarget(uint32_t target, uint32_t& index) {
	index = 0;
	switch (target) {
		case 0x08u: return IR::ExportTargetKind::MrtZ;
		case 0x09u: return IR::ExportTargetKind::Null;
		case 0x14u: return IR::ExportTargetKind::Primitive;
		default: break;
	}
	if (target <= 0x07u) {
		index = target;
		return IR::ExportTargetKind::Mrt;
	}
	if (target >= 0x0cu && target <= 0x0fu) {
		index = target - 0x0cu;
		return IR::ExportTargetKind::Position;
	}
	if (target >= 0x20u && target <= 0x3fu) {
		index = target - 0x20u;
		return IR::ExportTargetKind::Parameter;
	}
	return IR::ExportTargetKind::Unknown;
}

} // namespace

IR::ExportFlags Translator::AddExportInfo(const Decoder::Instruction& inst) {
	IR::ExportInfo info;
	info.kind        = ExportTargetKindFromTarget(inst.exp.target, info.index);
	info.target      = inst.exp.target;
	info.en          = inst.exp.en;
	info.done        = inst.exp.done;
	info.compr       = inst.exp.compr;
	info.vm          = inst.exp.vm;
	const auto index = static_cast<uint32_t>(program.export_info.size());
	program.export_info.push_back(info);
	return {.index = index, .pc = inst.pc};
}

void Translator::TranslateEmbeddedFetch(const Decoder::Instruction& inst, uint32_t attribute,
                                        uint32_t component_count,
                                        const ShaderBufferResource& resource) {
	const auto format = Format::GetFormatInfo(resource.Format());
	for (uint32_t component = 0; component < component_count; component++) {
		auto source = Format::FormattedSource {Format::FormattedSourceKind::Memory, component};
		if (inst.formatted && !inst.typed) {
			source = Format::ResolveFormattedSource(
			    format, GetDstSel(resource.DstSelXYZW(), component));
			if (source.kind == Format::FormattedSourceKind::Invalid) {
				EXIT("invalid formatted vertex input %u at pc 0x%08x", attribute, inst.pc);
			}
		}
		IR::Value value;
		if (source.kind == Format::FormattedSourceKind::Memory) {
			value = ir.Emit(IR::ValueOpcode::GetAttribute,
			                {IR::Value(attribute), IR::Value(source.component)});
			auto& required = program.info.vertex_fetch_components[attribute];
			required = static_cast<uint8_t>(std::max<uint32_t>(required, source.component + 1u));
		} else {
			value = IR::Value(Format::FormattedConstantBits(format, source.kind));
		}
		WriteOperand(OffsetOperand(inst.dst, component), value);
	}
}

void Translator::V_INTERP_P1_F32() {}

void Translator::V_INTERP_P2_F32(const Decoder::Instruction& inst) {
	const auto value = ir.Emit(IR::ValueOpcode::GetAttribute,
	                           {IR::Value(inst.src1.value), IR::Value(inst.src2.value)});
	WriteOperand(inst.dst, value);
}

void Translator::V_INTERP_MOV_F32(const Decoder::Instruction& inst) {
	if (inst.src0.value >= 3u) {
		EXIT("v_interp_mov_f32 mode %u is reserved at pc 0x%08x", inst.src0.value, inst.pc);
	}
	const auto value = ir.Emit(
	    IR::ValueOpcode::GetInterpolationParameter,
	    {IR::Value(inst.src1.value), IR::Value(inst.src2.value), IR::Value(inst.src0.value)});
	WriteOperand(inst.dst, value);
}

void Translator::EXP(const Decoder::Instruction& inst) {
	uint32_t index = 0;
	if (ExportTargetKindFromTarget(inst.exp.target, index) == IR::ExportTargetKind::Unknown) {
		EXIT("unsupported EXP target 0x%02x at pc 0x%08x", inst.exp.target, inst.pc);
	}
	std::array<IR::Value, 4> components {IR::Value(0u), IR::Value(0u), IR::Value(0u),
	                                     IR::Value(0u)};
	for (uint32_t source = 0; source < std::min(inst.src_count, 4u); source++) {
		components[source] = ReadRawU32(PlainOperand(SourceAt(inst, source)));
	}
	if (DebugLoopHeat().Active(program) && inst.exp.target < 8u && inst.exp.en != 0u) {
		// Keep the shader's own results alive, so its resources (and the recorded specializations
		// that index them) stay the same.
		for (uint32_t source = 0; source < std::min(inst.src_count, 4u); source++) {
			ir.Emit(IR::ValueOpcode::ReferenceU32, {components[source]});
		}
		// Band each counter as a float (F16 halves when the export is compressed).
		const auto band = [&](uint32_t counter, bool half) {
			const auto count =
			    ir.GetVectorReg(static_cast<IR::VectorReg>(LoopHeat::FirstRegister + counter));
			constexpr std::array<std::pair<uint32_t, float>, 5> bands {
			    {{1u, 0.0625f}, {16u, 0.25f}, {64u, 1.0f}, {256u, 4.0f}, {1024u, 32.0f}}};
			constexpr std::array<uint32_t, 5> half_bits {0x2c00u, 0x3400u, 0x3c00u, 0x4400u,
			                                             0x5000u};
			IR::U32 value(IR::Value(0u));
			for (size_t index = 0; index < bands.size(); index++) {
				const auto bits =
				    half ? half_bits[index] : std::bit_cast<uint32_t>(bands[index].second);
				const auto at_least = IR::U1(ir.Emit(IR::ValueOpcode::UGreaterThanEqual32,
				                                     {count, IR::Value(bands[index].first)}));
				value = ir.Select(at_least, IR::U32(IR::Value(bits)), value);
			}
			return value;
		};
		if (inst.exp.compr) {
			const auto shift = IR::U32(IR::Value(16u));
			components[0] = ir.BitwiseOr(band(0, true), ir.ShiftLeftLogical(band(1, true), shift));
			components[1] = ir.BitwiseOr(band(2, true), IR::U32(IR::Value(0x3c000000u)));
		} else {
			components = {band(0, false), band(1, false), band(2, false),
			              IR::Value(std::bit_cast<uint32_t>(1.0f))};
		}
	}
	const auto data = ir.Emit(IR::ValueOpcode::CompositeConstructU32x4,
	                          {components[0], components[1], components[2], components[3]});
	ir.Emit(IR::ValueOpcode::SetAttribute, {data, ir.GetExec()}, AddExportInfo(inst));
}

void Translator::EmitInterpolation(const Decoder::Instruction& inst) {
	switch (inst.opcode) {
		case Decoder::Opcode::V_INTERP_P1_F32: V_INTERP_P1_F32(); return;
		case Decoder::Opcode::V_INTERP_P2_F32: V_INTERP_P2_F32(inst); return;
		case Decoder::Opcode::V_INTERP_MOV_F32: V_INTERP_MOV_F32(inst); return;
		default: return FailMissingTranslation(inst);
	}
}

} // namespace Libs::Graphics::ShaderRecompiler::Frontend
