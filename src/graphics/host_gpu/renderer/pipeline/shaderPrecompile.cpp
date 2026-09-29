#include "graphics/host_gpu/renderer/pipeline/shaderPrecompile.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <xxhash.h>

namespace Libs::Graphics::ShaderPrecompile {
namespace {

constexpr std::array<uint8_t, 8> Magic {'K', 'Y', 'T', 'Y', 'S', 'H', 'D', 'R'};
constexpr uint32_t               FormatVersion  = 4; // 4: vertex buffer_index, pixel alpha_blend_source_remap.
constexpr uint32_t               MaxCodeWords   = 256u * 1024u;
constexpr uint32_t               MaxRecordBytes = 4u * 1024u * 1024u;
constexpr uint64_t               MaxFileBytes   = 256u * 1024u * 1024u;
constexpr uint32_t               MaxRecords     = 32768u;
constexpr uint32_t               MaxKeyBytes    = 4096u;

struct Encoder {
	std::vector<uint8_t> bytes;
	bool                 ok = true;

	void Word(uint32_t value) {
		for (uint32_t shift = 0; shift != 32; shift += 8) {
			bytes.push_back(static_cast<uint8_t>(value >> shift));
		}
	}
	void Wide(uint64_t value) {
		Word(static_cast<uint32_t>(value));
		Word(static_cast<uint32_t>(value >> 32u));
	}
	template <typename T>
	void Value(T& value) {
		if constexpr (std::is_array_v<T>) {
			for (auto& element: value)
				Value(element);
		} else if constexpr (std::is_same_v<T, float>) {
			Word(std::bit_cast<uint32_t>(value));
		} else {
			static_assert(std::is_integral_v<T> || std::is_enum_v<T>);
			static_assert(sizeof(T) <= sizeof(uint32_t));
			Word(static_cast<uint32_t>(value));
		}
	}
	template <typename... T>
	void operator()(T&... value) {
		(Value(value), ...);
	}
	void Words(std::vector<uint32_t>& words) {
		if (words.size() > MaxCodeWords) {
			ok = false;
			return;
		}
		Word(static_cast<uint32_t>(words.size()));
		for (auto word: words)
			Word(word);
	}
};

struct Decoder {
	std::span<const uint8_t> bytes;
	size_t                   offset = 0;
	bool                     ok     = true;

	[[nodiscard]] size_t Remaining() const { return bytes.size() - offset; }
	uint32_t             Word() {
		if (!ok || Remaining() < 4u) {
			ok = false;
			return 0;
		}
		uint32_t result = 0;
		for (uint32_t shift = 0; shift != 32; shift += 8) {
			result |= static_cast<uint32_t>(bytes[offset++]) << shift;
		}
		return result;
	}
	uint64_t Wide() {
		const auto low = Word();
		return low | (static_cast<uint64_t>(Word()) << 32u);
	}
	template <typename T>
	void Value(T& value) {
		if constexpr (std::is_array_v<T>) {
			for (auto& element: value)
				Value(element);
		} else {
			const auto bits = Word();
			if constexpr (std::is_same_v<T, bool>) {
				if (bits > 1u) ok = false;
				value = bits != 0;
			} else if constexpr (std::is_same_v<T, float>) {
				value = std::bit_cast<float>(bits);
			} else if constexpr (std::is_enum_v<T>) {
				value = static_cast<T>(bits);
				if (static_cast<uint32_t>(value) != bits) ok = false;
			} else if constexpr (std::is_signed_v<T>) {
				static_assert(sizeof(T) == sizeof(uint32_t));
				value = std::bit_cast<T>(bits);
			} else {
				static_assert(std::is_integral_v<T> && sizeof(T) <= sizeof(uint32_t));
				if (bits > std::numeric_limits<T>::max()) ok = false;
				value = static_cast<T>(bits);
			}
		}
	}
	template <typename... T>
	void operator()(T&... value) {
		(Value(value), ...);
	}
	void Words(std::vector<uint32_t>& words) {
		const auto count = Word();
		if (!ok || count > MaxCodeWords || count > Remaining() / 4u) {
			ok = false;
			return;
		}
		words.resize(count);
		for (auto& word: words)
			word = Word();
	}
};

template <typename Archive>
void Fields(Archive& a, ShaderWorkgroupInputInfo& i) {
	a(i.threads_num, i.lds_size_dwords, i.scratch_size_dwords, i.host_subgroup_size, i.wave_size);
}

template <typename Archive>
void Fields(Archive& a, ShaderVertexInputInfo& i) {
	a(i.logical_stage, i.resources_num, i.fetch_attrib_reg, i.fetch_buffer_reg, i.buffers_num,
	  i.wave_size, i.scratch_size_dwords, i.pa_cl_vs_out_cntl);
	if (i.resources_num < 0 || i.resources_num > ShaderVertexInputInfo::RES_MAX ||
	    i.buffers_num < 0 || i.buffers_num > ShaderVertexInputInfo::RES_MAX) {
		a.ok = false;
		return;
	}
	for (int n = 0; n < i.resources_num; ++n) {
		// Only descriptor shape participates in vertex translation. Never persist Base48 or
		// the runtime record count; restoring these fields to zero is deliberate.
		auto high = i.resources[n].fields[1] & 0xffff0000u;
		a(high, i.resources[n].fields[3]);
		i.resources[n].fields[0] = 0;
		i.resources[n].fields[1] = high & 0xffff0000u;
		i.resources[n].fields[2] = 0;
		auto& d                  = i.resources_dst[n];
		a(d.register_start, d.registers_num, d.attr_id, d.fetch_index, d.buffer_index);
	}
	for (int n = 0; n < i.buffers_num; ++n) {
		auto& b = i.buffers[n];
		a(b.stride, b.fetch_index);
		b.addr        = 0;
		b.num_records = 0;
	}
	a(i.clip_space.scale, i.clip_space.offset, i.clip_space.half_extent, i.clip_space.enabled);
	Fields(a, static_cast<ShaderWorkgroupInputInfo&>(i.mesh));
	a(i.mesh.input_primitive, i.mesh.primitives_per_group, i.mesh.vertices_per_group,
	  i.mesh.max_vertices, i.mesh.max_primitives, i.mesh.provoking_vertex);
	a(i.tess.input_control_points, i.tess.output_control_points, i.tess.ls_stride, i.tess.hs_stride,
	  i.tess.domain, i.tess.partitioning, i.tess.output_topology);
	a(i.fetch_external, i.fetch_embedded);
}

template <typename Archive>
void Fields(Archive& a, ShaderPixelInputInfo& i) {
	a(i.interpolator_settings, i.input_num, i.wave_size, i.ps_system_input_base,
	  i.custom_interpolation_mask, i.ps_perspective_center_vgpr, i.ps_perspective_centroid_vgpr,
	  i.target_output_mode);
	for (auto& mapping: i.target_export_mapping)
		a(mapping.packed);
	a(i.scratch_size_dwords, i.ps_pos_x, i.ps_pos_y, i.ps_pos_z, i.ps_pos_w, i.ps_front_face,
	  i.ps_ancillary, i.ps_no_perspective, i.ps_pixel_kill_enable, i.ps_depth_export_enable,
	  i.ps_sample_mask_export_enable, i.ps_sample_shading, i.dual_source_blending, i.ps_early_z,
	  i.ps_execute_on_noop, i.alpha_blend_source_remap);
}

template <typename Archive>
void Fields(Archive& a, ShaderComputeInputInfo& i) {
	Fields(a, static_cast<ShaderWorkgroupInputInfo&>(i));
	// Dispatch sizes are runtime-only; the module uses declared threads_num.
	a(i.group_id, i.dispatch_thread_dimensions, i.thread_ids_num, i.workgroup_register,
	  i.tg_size_en);
}

template <typename Archive>
void Specialization(Archive& a, ShaderRecompiler::IR::ResourceSpecialization& s) {
	uint32_t count = static_cast<uint32_t>(s.buffers.size());
	a(count);
	if (count > ShaderRecompiler::IR::ShaderInfo::MaxBuffers) {
		a.ok = false;
		return;
	}
	s.buffers.resize(count);
	for (auto& b: s.buffers)
		a(b.packed_stride, b.descriptor_format, b.descriptor_swizzle, b.zero_stride_oob);
	count = static_cast<uint32_t>(s.images.size());
	a(count);
	if (count > ShaderRecompiler::IR::ShaderInfo::MaxImages) {
		a.ok = false;
		return;
	}
	s.images.resize(count);
	for (auto& i: s.images) {
		a(i.numeric_class, i.dimension, i.mip_count, i.conversion_format, i.shader_swizzle,
		  i.indirect_root, i.indirect_mapping_offset, i.indirect_search_iterations, i.cube,
		  i.fmask);
	}
}

bool IsVertexStage(ShaderType stage) {
	return stage == ShaderType::Vertex || stage == ShaderType::Mesh || stage == ShaderType::Local ||
	       stage == ShaderType::TessellationControl || stage == ShaderType::TessellationEvaluation;
}

bool Valid(const PermutationRecord& r) {
	if (r.code.empty() || r.code.size() > MaxCodeWords || r.back_code.size() > MaxCodeWords ||
	    r.user_data_count > ShaderParams {}.user_data.size() || r.user_data_base > 64u ||
	    (r.wave_size != 32u && r.wave_size != 64u) ||
	    (r.push_data_start_dword > ShaderRecompiler::IR::PushData::DwordCount &&
	     r.push_data_start_dword != ShaderRecompiler::IR::PushData::NoStart))
		return false;
	if (r.specialization.buffers.size() > ShaderRecompiler::IR::ShaderInfo::MaxBuffers ||
	    r.specialization.images.size() > ShaderRecompiler::IR::ShaderInfo::MaxImages)
		return false;
	for (const auto& b: r.specialization.buffers) {
		if (static_cast<uint32_t>(b.descriptor_format) > 0x1ffu || b.descriptor_swizzle > 0xfffu)
			return false;
	}
	for (const auto& i: r.specialization.images) {
		if (static_cast<uint32_t>(i.numeric_class) > 3u ||
		    static_cast<uint32_t>(i.dimension) > 7u || i.mip_count == 0 || i.mip_count > 16u ||
		    static_cast<uint32_t>(i.conversion_format) > 0x1ffu || i.shader_swizzle > 0xfffu ||
		    (i.indirect_root != ShaderRecompiler::IR::ImageResource::NoIndirectImage &&
		     i.indirect_root >= r.specialization.images.size()) ||
		    i.indirect_search_iterations > 32u)
			return false;
	}
	if (const auto* i = std::get_if<ShaderVertexInputInfo>(&r.info)) {
		if (!IsVertexStage(r.stage) || i->logical_stage != r.stage || i->resources_num < 0 ||
		    i->resources_num > ShaderVertexInputInfo::RES_MAX || i->buffers_num < 0 ||
		    i->buffers_num > ShaderVertexInputInfo::RES_MAX ||
		    (i->wave_size != 32u && i->wave_size != 64u))
			return false;
		const auto expected_wave = r.stage == ShaderType::Mesh                  ? i->mesh.wave_size
		                           : r.stage == ShaderType::TessellationControl ? 64u
		                                                                        : i->wave_size;
		if (r.wave_size != expected_wave) return false;
		for (int n = 0; n < i->resources_num; ++n) {
			const auto& d = i->resources_dst[n];
			if (d.register_start < 0 || d.register_start > 255 || d.registers_num < 0 ||
			    d.registers_num > 4 || d.attr_id < -1 || d.attr_id >= 32 || d.buffer_index < 0 ||
			    d.buffer_index >= i->buffers_num)
				return false;
		}
		return true;
	}
	if (const auto* i = std::get_if<ShaderPixelInputInfo>(&r.info)) {
		return r.stage == ShaderType::Pixel && i->input_num <= 32u && i->wave_size == r.wave_size &&
		       i->ps_system_input_base <= 255u &&
		       (i->ps_perspective_center_vgpr == UINT32_MAX ||
		        i->ps_perspective_center_vgpr < 255u) &&
		       (i->ps_perspective_centroid_vgpr == UINT32_MAX ||
		        i->ps_perspective_centroid_vgpr < 255u);
	}
	const auto& i = std::get<ShaderComputeInputInfo>(r.info);
	return r.stage == ShaderType::Compute && i.wave_size == r.wave_size &&
	       (i.host_subgroup_size == 32u || i.host_subgroup_size == 64u) && i.thread_ids_num >= 0 &&
	       i.thread_ids_num <= 3 && i.workgroup_register >= 0 && i.workgroup_register <= 104;
}

bool Write(std::ofstream& out, std::span<const uint8_t> bytes) {
	out.write(reinterpret_cast<const char*>(bytes.data()),
	          static_cast<std::streamsize>(bytes.size()));
	return static_cast<bool>(out);
}

} // namespace

bool Encode(const PermutationRecord& record, std::vector<uint8_t>& bytes) {
	bytes.clear();
	if (!Valid(record)) return false;
	// Fields is shared by reader and writer. The writer changes only this temporary's
	// runtime address fields, ensuring those cannot accidentally enter the persistent format.
	auto    r = record;
	Encoder a;
	a(r.stage);
	a.Wide(r.hash);
	a(r.push_data_start_dword, r.user_data_base, r.user_data_count, r.wave_size);
	a.Words(r.code);
	a.Words(r.back_code);
	Specialization(a, r.specialization);
	std::visit([&](auto& i) { Fields(a, i); }, r.info);
	if (!a.ok || a.bytes.size() > MaxRecordBytes) return false;
	bytes = std::move(a.bytes);
	return true;
}

bool Decode(std::span<const uint8_t> bytes, PermutationRecord& record) {
	if (bytes.empty() || bytes.size() > MaxRecordBytes) return false;
	Decoder           a {bytes};
	PermutationRecord r;
	a(r.stage);
	r.hash = a.Wide();
	a(r.push_data_start_dword, r.user_data_base, r.user_data_count, r.wave_size);
	a.Words(r.code);
	a.Words(r.back_code);
	Specialization(a, r.specialization);
	if (IsVertexStage(r.stage))
		r.info = ShaderVertexInputInfo {};
	else if (r.stage == ShaderType::Pixel)
		r.info = ShaderPixelInputInfo {};
	else if (r.stage == ShaderType::Compute)
		r.info = ShaderComputeInputInfo {};
	else
		return false;
	std::visit([&](auto& i) { Fields(a, i); }, r.info);
	if (!a.ok || a.Remaining() != 0 || !Valid(r)) return false;
	record = std::move(r);
	return true;
}

std::vector<PermutationRecord> Journal::Open(const std::filesystem::path& path,
                                             std::string_view             key) {
	Close();
	m_out.clear();
	m_count = 0;
	m_bytes = 0;
	std::vector<PermutationRecord> records;
	if (path.empty() || key.empty() || key.size() > MaxKeyBytes) return records;

	Encoder header;
	header.bytes.insert(header.bytes.end(), Magic.begin(), Magic.end());
	header.Word(FormatVersion);
	header.Word(static_cast<uint32_t>(key.size()));
	header.bytes.insert(header.bytes.end(), key.begin(), key.end());
	std::error_code ec;
	const auto      size = std::filesystem::file_size(path, ec);
	if (ec && ec != std::errc::no_such_file_or_directory) return records;
	bool   compatible = false;
	size_t accepted   = 0;
	if (!ec && size > MaxFileBytes) return records;
	if (!ec && size >= header.bytes.size()) {
		std::ifstream in(path, std::ios::binary);
		if (!in) return records;
		std::vector<uint8_t> blob(static_cast<size_t>(size));
		in.read(reinterpret_cast<char*>(blob.data()), static_cast<std::streamsize>(blob.size()));
		if (!in) return records;
		compatible = std::equal(header.bytes.begin(), header.bytes.end(), blob.begin());
		if (compatible) {
			Decoder envelope {blob, header.bytes.size()};
			accepted = envelope.offset;
			while (envelope.Remaining() != 0 && records.size() < MaxRecords) {
				const auto bytes    = envelope.Word();
				const auto checksum = envelope.Wide();
				if (!envelope.ok || bytes == 0 || bytes > MaxRecordBytes ||
				    bytes > envelope.Remaining())
					break;
				const auto        body = std::span(blob).subspan(envelope.offset, bytes);
				PermutationRecord record;
				if (XXH3_64bits(body.data(), body.size()) != checksum || !Decode(body, record))
					break;
				records.push_back(std::move(record));
				envelope.offset += bytes;
				accepted = envelope.offset;
			}
		}
	}
	if (compatible) {
		if (accepted != size) {
			std::filesystem::resize_file(path, accepted, ec);
			if (ec) return records;
		}
		m_out.open(path, std::ios::binary | std::ios::app);
		m_bytes = accepted;
	} else {
		ec.clear();
		if (!path.parent_path().empty())
			std::filesystem::create_directories(path.parent_path(), ec);
		if (ec) return records;
		m_out.open(path, std::ios::binary | std::ios::trunc);
		if (!m_out || !Write(m_out, header.bytes)) {
			Close();
			return records;
		}
		m_out.flush();
		if (!m_out) {
			Close();
			return records;
		}
		m_bytes = header.bytes.size();
	}
	m_count = records.size();
	return records;
}

bool Journal::Append(const PermutationRecord& record) {
	if (!IsOpen() || m_count >= MaxRecords) return false;
	std::vector<uint8_t> body;
	if (!Encode(record, body)) return false;
	Encoder envelope;
	envelope.Word(static_cast<uint32_t>(body.size()));
	envelope.Wide(XXH3_64bits(body.data(), body.size()));
	const auto bytes = envelope.bytes.size() + body.size();
	if (bytes > MaxFileBytes - m_bytes) return false;
	if (!Write(m_out, envelope.bytes) || !Write(m_out, body)) {
		Close();
		return false;
	}
	// Preserve complete records across a later crash. On an I/O failure stop appending; the
	// next Open can repair the tail instead of splicing subsequent records into damaged data.
	m_out.flush();
	if (!m_out) {
		Close();
		return false;
	}
	m_bytes += bytes;
	++m_count;
	return true;
}

void Journal::Close() {
	if (m_out.is_open()) {
		m_out.flush();
		m_out.close();
	}
}

} // namespace Libs::Graphics::ShaderPrecompile
