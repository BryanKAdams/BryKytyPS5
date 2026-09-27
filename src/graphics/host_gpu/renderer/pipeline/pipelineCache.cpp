#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineLibrary.h"
#include "graphics/host_gpu/renderer/pipeline/shaderPrecompile.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/shaderCompiler.h"
#include "kernel/memory.h"
#include "kytyGitVersion.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <future>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <spirv-tools/libspirv.hpp>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

// The library cache key a prefetched compute pipeline compiles under.
std::string ComputePrefetchKey(uint64_t program_id) {
	std::string key(1, 'C');
	key.append(reinterpret_cast<const char*>(&program_id), sizeof(program_id));
	return key;
}

void DestroyPipelineObjects(const GraphicContext& graphics, const PipelineCache::Pipeline& pipeline) {
	if (pipeline.pipeline != nullptr) {
		graphics.device.destroyPipeline(pipeline.pipeline, nullptr);
	}
	graphics.device.destroyPipelineLayout(pipeline.pipeline_layout, nullptr);
	graphics.device.destroyDescriptorSetLayout(pipeline.descriptor_set_layout, nullptr);
	if (pipeline.pixel_set_layout != nullptr) {
		graphics.device.destroyDescriptorSetLayout(pipeline.pixel_set_layout, nullptr);
	}
}

vk::PolygonMode ResolvePolygonMode(const HW::ModeControl& mode, bool cull_front, bool cull_back) {
	// CxPrimitiveSetup::PolygonMode disables both per-face modes when it is zero.
	if (mode.poly_mode == 0) {
		return vk::PolygonMode::eFill;
	}
	EXIT_NOT_IMPLEMENTED(mode.poly_mode != 1);
	if (cull_front && cull_back) {
		return vk::PolygonMode::eFill;
	}
	if (!cull_front && !cull_back && mode.polymode_front_ptype != mode.polymode_back_ptype) {
		EXIT("Pipeline: different polygon modes for two visible faces are unsupported\n");
	}
	// Vulkan has one polygon mode. A culled face does not constrain that mode.
	const auto polygon_mode = cull_front ? mode.polymode_back_ptype : mode.polymode_front_ptype;
	switch (polygon_mode) {
		case 0: return vk::PolygonMode::ePoint;
		case 1: return vk::PolygonMode::eLine;
		case 2: return vk::PolygonMode::eFill;
		default: EXIT("Pipeline: invalid polygon mode %u\n", polygon_mode);
	}
}

std::string DriverCacheSignature(const vk::PhysicalDeviceProperties& properties) {
	constexpr char hex[] = "0123456789abcdef";
	std::string    uuid(VK_UUID_SIZE * 2, '0');
	for (size_t i = 0; i < VK_UUID_SIZE; i++) {
		uuid[i * 2]     = hex[properties.pipelineCacheUUID[i] >> 4u];
		uuid[i * 2 + 1] = hex[properties.pipelineCacheUUID[i] & 0xfu];
	}
	return fmt::format("KytyPC3:{}:{:08x}:{:08x}:{:08x}:{}:opt={}\n", KYTY_SHADER_CACHE_KEY,
	                   properties.vendorID, properties.deviceID, properties.driverVersion, uuid,
	                   static_cast<uint32_t>(Config::GetShaderOptimizationType()));
}

std::string PipelineCacheTitleId() {
	std::string title_id;
	if ((!Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id) || title_id.empty()) &&
	    (!Loader::SystemContentParamSfoGetString("CONTENT_ID", &title_id) || title_id.empty())) {
		return {};
	}
	if (!std::ranges::all_of(title_id, [](unsigned char c) {
		    return std::isalnum(c) != 0 || c == '-' || c == '_';
	    })) {
		return {};
	}
	return title_id;
}

template <typename... Args>
void PipelineCacheLog(fmt::format_string<Args...> format, Args&&... args) {
	auto message = fmt::format(format, std::forward<Args>(args)...);
	message += '\n';
	Log::WriteToConsoleAndLog(message);
}

bool ReadShaderGuestMemory(void*, uint64_t address, std::span<uint32_t> values) {
	return !values.empty() &&
	       Libs::LibKernel::Memory::TryReadGpuCleanBacking(address, values.data(), values.size_bytes());
}

// Per-draw resource tables can share a tracker page with GPU-written data, which protects the
// whole page. Reading their clean bytes from the backing avoids a fault and a GPU wait; bytes
// the GPU did write still fault and read back.
bool ReadShaderGuestMemoryOnGpuThread(void*, uint64_t address, std::span<uint32_t> values) {
	Libs::LibKernel::Memory::ReadGuestOnGpuThread(address, values.data(), values.size_bytes());
	return true;
}

void DumpShaderSpirv(const char* stage_name, uint64_t shader_hash,
                     const std::vector<uint32_t>& spirv) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	static std::atomic_int id = 0;
	const auto path = Config::GetShaderLogFolder() / fmt::format("{:04d}_new_shader_{}_{:016x}.spv",
	                                                             id++, stage_name, shader_hash);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(spirv.data(), spirv.size() * sizeof(uint32_t));
}

void DumpShaderOriginal(const char* stage_name, uint64_t shader_hash,
                        std::span<const uint32_t> code, const std::string& decoded_dump) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	EXIT_IF(code.empty());
	static std::atomic_int id = 0;
	const auto base = Config::GetShaderLogFolder() / "original" /
	                  fmt::format("{:04d}_new_shader_{}_{:016x}", id++, stage_name, shader_hash);
	Common::File::CreateDirectories(base.parent_path());
	for (const auto& [suffix, data, size]: {
	         std::tuple {".bin", static_cast<const void*>(code.data()), code.size_bytes()},
	         std::tuple {".rdna2", static_cast<const void*>(decoded_dump.data()),
	                     decoded_dump.size()},
	     }) {
		if (size == 0) {
			continue;
		}
		auto path = base;
		path += suffix;
		Common::File file(path);
		if (file.IsInvalid()) {
			const auto path_text = Common::PathToString(path);
			LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		} else {
			file.Write(data, size);
		}
	}
}

bool ValidateShaderSpirv(const char* label, uint64_t shader_hash,
                         const std::vector<uint32_t>& spirv) {
	if (!Config::ShaderValidationEnabled()) {
		return true;
	}
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          messages;
	tools.SetMessageConsumer([&messages](spv_message_level_t, const char*,
	                                     const spv_position_t& position, const char* message) {
		messages += fmt::format("{}: {} ({}) {}\n", static_cast<int>(position.line),
		                        static_cast<int>(position.column), static_cast<int>(position.index),
		                        message);
	});
	if (tools.Validate(spirv)) {
		return true;
	}
	spvtools::SpirvTools disassembler(SPV_ENV_VULKAN_1_2);
	std::string          text;
	disassembler.Disassemble(spirv, &text,
	                         static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COMMENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_INDENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COLOR));
	LOGF_COLOR(Log::Color::BrightRed, "%s SPIR-V validation failed hash=0x%016" PRIx64 ":\n%s",
	           label, shader_hash, messages.c_str());
	LOGF("%s\n", text.c_str());
	return false;
}

} // namespace

// KYTY_PERMUTATION_LOG=1 (diagnostic): describe why each runtime-compiled permutation is new.
[[nodiscard]] static bool PermutationLogEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_PERMUTATION_LOG");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

// The first few fields in which two specializations differ, e.g. "img[2].mip_count 1->12".
[[nodiscard]] static std::string
SpecializationDiff(const ShaderRecompiler::IR::ResourceSpecialization& a,
                   const ShaderRecompiler::IR::ResourceSpecialization& b) {
	std::string text;
	size_t      shown = 0;
	const auto  add   = [&](const char* kind, size_t index, const char* field, uint64_t from,
                         uint64_t to) {
        if (from != to && shown++ < 6) {
            text += fmt::format(" {}[{}].{} {}->{}", kind, index, field, from, to);
        }
	};
	if (a.buffers.size() != b.buffers.size() || a.images.size() != b.images.size()) {
		return fmt::format(" counts buffers {}->{} images {}->{}", a.buffers.size(),
		                   b.buffers.size(), a.images.size(), b.images.size());
	}
	for (size_t i = 0; i < a.buffers.size(); i++) {
		const auto& x = a.buffers[i];
		const auto& y = b.buffers[i];
		add("buf", i, "stride", x.packed_stride, y.packed_stride);
		add("buf", i, "format", static_cast<uint64_t>(x.descriptor_format),
		    static_cast<uint64_t>(y.descriptor_format));
		add("buf", i, "swizzle", x.descriptor_swizzle, y.descriptor_swizzle);
		add("buf", i, "oob", static_cast<uint64_t>(x.zero_stride_oob),
		    static_cast<uint64_t>(y.zero_stride_oob));
	}
	for (size_t i = 0; i < a.images.size(); i++) {
		const auto& x = a.images[i];
		const auto& y = b.images[i];
		add("img", i, "class", static_cast<uint64_t>(x.numeric_class),
		    static_cast<uint64_t>(y.numeric_class));
		add("img", i, "dim", static_cast<uint64_t>(x.dimension), static_cast<uint64_t>(y.dimension));
		add("img", i, "mip_count", x.mip_count, y.mip_count);
		add("img", i, "conversion", static_cast<uint64_t>(x.conversion_format),
		    static_cast<uint64_t>(y.conversion_format));
		add("img", i, "swizzle", x.shader_swizzle, y.shader_swizzle);
		add("img", i, "indirect_root", x.indirect_root, y.indirect_root);
		add("img", i, "indirect_offset", x.indirect_mapping_offset, y.indirect_mapping_offset);
		add("img", i, "indirect_iterations", x.indirect_search_iterations,
		    y.indirect_search_iterations);
		add("img", i, "cube", x.cube ? 1u : 0u, y.cube ? 1u : 0u);
		add("img", i, "fmask", x.fmask ? 1u : 0u, y.fmask ? 1u : 0u);
	}
	return text.empty() ? " (same specialization)" : text;
}

struct PipelineCache::ProgramCache {
	struct ProgramKey {
		ShaderType            stage           = ShaderType::Unknown;
		uint64_t              hash            = 0;
		uint32_t              user_data_count = 0;
		uint32_t              code_size       = 0;
		std::vector<uint32_t> static_state;

		bool operator==(const ProgramKey&) const = default;
	};

	struct Permutation {
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		ShaderRecompiler::IR::CompiledShaderInfo     program;
		ShaderProgram                                handle;
	};

	// A permutation's shader module and info, compiled on any thread.
	struct CompiledModule {
		ShaderRecompiler::IR::CompiledShaderInfo program;
		vk::ShaderModule                         module      = nullptr;
		size_t                                   spirv_words = 0;
	};

	// A permutation compiling on a worker thread.
	struct PendingPermutation {
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		uint32_t                                     push_data_cursor = 0;
		std::future<CompiledModule>                  compiled;
	};

	struct SourceEntry {
		explicit SourceEntry(ShaderRecompiler::IR::ResourcePlan plan)
		    : resource_plan(std::move(plan)) {
			permutations.reserve(8);
		}

		ShaderRecompiler::IR::ResourcePlan           resource_plan;
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		ShaderRecompiler::IR::MaterializationMemo    memo;
		std::vector<Permutation>                    permutations;
		// Permutations only grow and never repeat a (push data start, specialization) pair, so
		// the last hit stays valid while the specialization and push data cursor are unchanged.
		uint32_t                                    last_permutation = UINT32_MAX;
		uint32_t                                    last_push_cursor = 0;
		bool                                        skip_dispatch = false;
		std::vector<PendingPermutation>             pending;
	};

	struct ProgramKeyHash {
		std::size_t operator()(const ProgramKey& key) const {
			std::size_t hash = static_cast<std::size_t>(key.stage);
			PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash));
			if constexpr (sizeof(std::size_t) < sizeof(uint64_t)) {
				PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash >> 32u));
			}
			PipelineKeyHash::Mix(hash, key.user_data_count);
			PipelineKeyHash::Mix(hash, key.code_size);
			PipelineKeyHash::Mix(hash, key.static_state.size());
			// Bucket same-shape static variants by source. ProgramKey equality performs the one
			// exact state comparison needed on a stable hit without hashing up to 429 words first.
			return hash;
		}
	};

	static constexpr std::size_t MaxStaticKeyWords = 13 + ShaderVertexInputInfo::RES_MAX * 13;

	static const char* StageName(ShaderType stage) {
		switch (stage) {
			case ShaderType::Vertex: return "vs";
			case ShaderType::Mesh: return "ms";
			case ShaderType::Local: return "ls";
			case ShaderType::TessellationControl: return "hs";
			case ShaderType::TessellationEvaluation: return "ds";
			case ShaderType::Pixel: return "ps";
			case ShaderType::Compute: return "cs";
			default: EXIT("invalid pipeline shader stage\n");
		}
		return nullptr;
	}

	// Compiles a translated permutation to its SPIR-V module. Reads only its arguments, so it may
	// run on a worker thread.
	static CompiledModule CompileModule(vk::Device device, std::span<const uint32_t> code,
	                                    const ShaderRecompiler::CompileOptions&             options,
	                                    ShaderRecompiler::TranslateResult                   translated,
	                                    const ShaderRecompiler::IR::ResourceSpecialization& specialization,
	                                    uint32_t push_data_start_dword) {
		const char* stage_name = StageName(options.stage);
		auto result = ShaderRecompiler::CompileProgram(std::move(translated), options,
		                                               specialization, push_data_start_dword);
		DumpShaderOriginal(stage_name, options.shader_hash, code, result.decoded_dump);
		if (!ValidateShaderSpirv(options.dump_label, options.shader_hash, result.spirv)) {
			DumpShaderSpirv(stage_name, options.shader_hash, result.spirv);
			EXIT("%s failed hash=0x%016" PRIx64 ": SPIR-V validation failed\n", options.dump_label,
			     options.shader_hash);
		}
		DumpShaderSpirv(stage_name, options.shader_hash, result.spirv);

		const auto module = CompileSPV(result.spirv, device);
		EXIT_IF(module == nullptr);
		return {
		    .program     = std::move(result.program).TakeCompiledInfo(),
		    .module      = module,
		    .spirv_words = result.spirv.size(),
		};
	}

	// Numbers a compiled permutation.
	Permutation MakePermutation(const ShaderRecompiler::CompileOptions&      options,
	                            ShaderRecompiler::IR::ResourceSpecialization specialization,
	                            CompiledModule                               compiled) {
		if (PermutationLogEnabled()) [[unlikely]] {
			std::printf("spirv: id=%llu %s hash=%016llx words=%zu\n",
			            static_cast<unsigned long long>(next_shader_id + 1), StageName(options.stage),
			            static_cast<unsigned long long>(options.shader_hash), compiled.spirv_words);
		}
		if (options.dump_ir) {
			LOGF("%s SPIR-V words=%" PRIu64 " wave_size=%u\n", options.dump_label,
			     static_cast<uint64_t>(compiled.spirv_words), options.wave_size);
		}
		return {
		    .specialization = std::move(specialization),
		    .program        = std::move(compiled.program),
		    .handle         = {.id = ++next_shader_id, .module = compiled.module},
		};
	}

	Permutation CompilePermutation(const ShaderParams&                          params,
	                               const ShaderRecompiler::CompileOptions&      options,
	                               ShaderRecompiler::TranslateResult            translated,
	                               ShaderRecompiler::IR::ResourceSpecialization specialization,
	                               uint32_t push_data_start_dword) {
		auto compiled = CompileModule(device, params.code, options, std::move(translated),
		                              specialization, push_data_start_dword);
		return MakePermutation(options, std::move(specialization), std::move(compiled));
	}

	template <typename InputInfo>
	static ShaderRecompiler::CompileOptions MakeOptions(ShaderType stage, uint64_t hash,
	                                                    std::span<const uint32_t> user_data,
	                                                    std::span<const uint32_t> back_code,
	                                                    InputInfo&                input_info) {
		ShaderStageInputInfo stage_input {};
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage_input.vertex = &input_info;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage_input.pixel = &input_info;
		} else {
			stage_input.compute = &input_info;
		}
		const char* label = nullptr;
		switch (stage) {
			case ShaderType::Vertex: label = "ShaderRecompiler VS"; break;
			case ShaderType::Mesh: label = "ShaderRecompiler MS"; break;
			case ShaderType::Local: label = "ShaderRecompiler LS"; break;
			case ShaderType::TessellationControl: label = "ShaderRecompiler HS"; break;
			case ShaderType::TessellationEvaluation: label = "ShaderRecompiler DS"; break;
			case ShaderType::Pixel: label = "ShaderRecompiler PS"; break;
			case ShaderType::Compute: label = "ShaderRecompiler CS"; break;
			default: EXIT("invalid pipeline shader stage\n");
		}
		ShaderRecompiler::CompileOptions options;
		options.stage       = stage;
		options.shader_hash = hash;
		options.user_data   = user_data;
		options.back_code   = back_code;
		options.dump_ir     = Config::GetShaderLogDirection() != Config::LogDirection::Silent;
		options.early_dump  = options.dump_ir;
		options.dump_label  = label;
		options.input_info  = stage_input;

		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			options.user_data_base = 8;
			options.wave_size      = input_info.wave_size;
			if (stage == ShaderType::Mesh || stage == ShaderType::TessellationControl) {
				options.user_data_base = 0;
				options.wave_size = stage == ShaderType::Mesh ? input_info.mesh.wave_size : 64u;
			}
		} else {
			options.wave_size = input_info.wave_size;
		}
		return options;
	}

	// A background translation's own copies of its inputs; `options` points into them, so the
	// job stays where it was allocated.
	template <typename InputInfo>
	struct TranslationInput {
		std::vector<uint32_t>            code;
		std::vector<uint32_t>            back_code;
		std::array<uint32_t, 40>         user_data {};
		InputInfo                        input_info;
		ShaderRecompiler::CompileOptions options;
	};

	template <typename InputInfo>
	static std::unique_ptr<TranslationInput<InputInfo>>
	CopyTranslationInput(ShaderType stage, const ShaderParams& params, const InputInfo& input_info) {
		static_assert(std::tuple_size_v<decltype(params.user_data)> == 40);
		auto input = std::make_unique<TranslationInput<InputInfo>>();
		input->code.assign(params.code.begin(), params.code.end());
		input->back_code.assign(params.back_code.begin(), params.back_code.end());
		input->user_data  = params.user_data;
		input->input_info = input_info;
		input->input_info.stage = {};
		input->options = MakeOptions(stage, params.hash,
		                             std::span(input->user_data).first(params.user_data_count),
		                             input->back_code, input->input_info);
		// Worker threads never write the shader log.
		input->options.dump_ir    = false;
		input->options.early_dump = false;
		return input;
	}

	// Posts a translation or module compile to the worker threads, counted in `jobs` until its
	// result is set.
	template <typename Job>
	void PostJob(Job&& job, bool urgent) {
		jobs->in_flight.fetch_add(1, std::memory_order_relaxed);
		const bool posted = workers->Post(
		    [counts = jobs, job = std::forward<Job>(job)]() mutable {
			    job();
			    counts->finished.fetch_add(1, std::memory_order_relaxed);
			    counts->in_flight.fetch_sub(1, std::memory_order_release);
		    },
		    urgent);
		if (!posted) {
			jobs->in_flight.fetch_sub(1, std::memory_order_relaxed);
		}
	}

	template <typename T>
	[[nodiscard]] static bool IsReady(const std::future<T>& future) {
		return future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
	}

	// Null when the job was dropped (the worker threads stopped) or failed.
	template <typename T>
	[[nodiscard]] static std::optional<T> TakeResult(std::future<T>& future) {
		try {
			return future.get();
		} catch (const std::exception&) {
			return std::nullopt;
		}
	}

	// Whether the shader stores to buffers, writes images or uses GDS, found by decoding only.
	// Buffer atomics do not count (see StoresData in renderDraw.cpp).
	bool StoresData(ShaderType stage, const ShaderParams& params) {
		if (const auto known = stores_data.find(params.hash); known != stores_data.end()) {
			return known->second;
		}
		namespace Decoder = ShaderRecompiler::Decoder;
		const auto stores = [](const Decoder::Program& program) {
			return std::ranges::any_of(program.instructions, [](const Decoder::Instruction& inst) {
				if (inst.family == Decoder::Family::DS && inst.gds) {
					return true;
				}
				const auto name = magic_enum::enum_name(inst.opcode);
				return name.starts_with("BUFFER_STORE") || name.starts_with("TBUFFER_STORE") ||
				       name.starts_with("FLAT_STORE") || name.starts_with("GLOBAL_STORE") ||
				       name.starts_with("IMAGE_STORE") || name.starts_with("IMAGE_ATOMIC");
			});
		};
		// Decoded as TranslateProgram decodes it.
		Decoder::Program front;
		if (!params.back_code.empty() || stage == ShaderType::Local) {
			front = Decoder::DecodeFrontProgram(params.code);
		} else {
			Decoder::DecodeProgram(params.code, front);
		}
		bool result = front.has_bvh || stores(front);
		if (!result && !params.back_code.empty()) {
			Decoder::Program back;
			Decoder::DecodeProgram(params.back_code, back);
			result = back.has_bvh || stores(back);
		}
		stores_data.emplace(params.hash, result);
		return result;
	}

	// Starts translating a source nothing has translated or queued yet (ProgramWait::Prefetch for
	// a stage whose push data position is unknown until an earlier stage compiles).
	template <typename InputInfo>
	void QueueSourceTranslation(const ShaderParams& params, const InputInfo& input_info,
	                            ShaderType stage) {
		if (workers == nullptr ||
		    Config::GetShaderLogDirection() != Config::LogDirection::Silent) {
			return;
		}
		lookup_key.stage           = stage;
		lookup_key.hash            = params.hash;
		lookup_key.user_data_count = params.user_data_count;
		lookup_key.code_size       = static_cast<uint32_t>(params.code.size());
		BuildStageStaticKey(input_info, lookup_key.static_state);
		if (programs.contains(lookup_key) || pending_sources.contains(lookup_key)) {
			return;
		}
		std::promise<ShaderRecompiler::TranslateResult> promise;
		pending_sources.emplace(lookup_key, promise.get_future());
		PostJob(
		    [input = CopyTranslationInput(stage, params, input_info),
		     promise = std::move(promise)]() mutable {
			    promise.set_value(ShaderRecompiler::TranslateProgram(input->code, input->options));
		    },
		    false);
	}

	// With ProgramWait::Defer (a draw, with asynchronous pipelines) or ProgramWait::Prefetch (a
	// look-ahead prediction), a permutation not compiled yet is translated and compiled on the
	// worker threads, and the result is empty with `*pending` set; the same lookup later picks
	// it up. Defer jobs go ahead of queued prefetches.
	template <typename InputInfo>
	ShaderProgram Get(const ShaderParams& params, InputInfo& input_info, uint32_t& push_data_cursor,
	                  ProgramWait wait = ProgramWait::Wait, bool* pending = nullptr) {
		ShaderType stage;
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage = input_info.logical_stage;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage = ShaderType::Pixel;
		} else {
			static_assert(std::is_same_v<InputInfo, ShaderComputeInputInfo>);
			stage = ShaderType::Compute;
		}
		const bool background = wait != ProgramWait::Wait && workers != nullptr &&
		                        Config::GetShaderLogDirection() == Config::LogDirection::Silent;
		const bool urgent     = wait == ProgramWait::Defer;

		const auto user_data = std::span(params.user_data).first(params.user_data_count);
		lookup_key.stage           = stage;
		lookup_key.hash            = params.hash;
		lookup_key.user_data_count = params.user_data_count;
		lookup_key.code_size       = static_cast<uint32_t>(params.code.size());
		BuildStageStaticKey(input_info, lookup_key.static_state);
		auto                                         entry = programs.find(lookup_key);
		if (entry != programs.end() && entry->second.skip_dispatch) {
			return {};
		}
		const ShaderRecompiler::IR::SrtRuntime       runtime {
		    .user_data                  = user_data,
		    .shader_base                = params.Base(),
		    .read_memory                = ReadShaderGuestMemoryOnGpuThread,
		    .read_specialization_memory = ReadShaderGuestMemory,
		    // TryReadGpuCleanBacking fails for a range when any byte is GPU-dirty or unbacked.
		    .specialization_block_reads = true,
		};
		if (entry != programs.end()) {
			auto& source = entry->second;
			EXIT_IF(!ShaderRecompiler::IR::MaterializeResources(
			    source.resource_plan, runtime, source.resources, source.specialization,
			    &source.memo));
			if (stats_enabled) {
				CountRefresh(source);
			}
			const auto matches = [&](const Permutation& candidate) {
				const auto& layout = candidate.program.bindings;
				return layout.push_data_start_dword ==
				           ShaderRecompiler::IR::PushData::StartFor(push_data_cursor,
				                                                    layout.ShaderDataDwords()) &&
				       candidate.specialization == source.specialization;
			};
			auto index = source.last_permutation;
			if (index >= source.permutations.size() ||
			    ((!source.memo.reused || source.last_push_cursor != push_data_cursor) &&
			     !matches(source.permutations[index]))) {
				index = static_cast<uint32_t>(
				    std::ranges::find_if(source.permutations, matches) -
				    source.permutations.begin());
			}
			if (index < source.permutations.size()) {
				auto& permutation       = source.permutations[index];
				source.last_permutation = index;
				source.last_push_cursor = push_data_cursor;
				input_info.stage = {.program = &permutation.program, .resources = &source.resources};
				permutation.program.bindings.AdvancePushData(push_data_cursor);
				return permutation.handle;
			}
		}

		const auto defer = [&] {
			if (pending != nullptr) {
				*pending = true;
			}
			// The refresh above may have changed the specialization without a permutation to match
			// it, so the next lookup must search instead of trusting the last hit.
			if (entry != programs.end()) {
				entry->second.last_permutation = UINT32_MAX;
			}
			return ShaderProgram {};
		};
		auto options = MakeOptions(stage, params.hash, user_data, params.back_code, input_info);
		std::optional<ShaderRecompiler::TranslateResult> translated;
		if (entry == programs.end()) {
			// A new source: translate it first.
			if (auto queued = pending_sources.find(lookup_key); queued != pending_sources.end()) {
				if (background && !IsReady(queued->second)) {
					return defer();
				}
				translated = TakeResult(queued->second);
				pending_sources.erase(queued);
			} else {
				if (PermutationLogEnabled()) [[unlikely]] {
					LogNewPermutation(stage, params.hash, entry, push_data_cursor);
				}
				if (background) {
					std::promise<ShaderRecompiler::TranslateResult> promise;
					pending_sources.emplace(lookup_key, promise.get_future());
					PostJob(
					    [input   = CopyTranslationInput(stage, params, input_info),
					     promise = std::move(promise)]() mutable {
						    promise.set_value(
						        ShaderRecompiler::TranslateProgram(input->code, input->options));
					    },
					    urgent);
					return defer();
				}
			}
			if (!translated) {
				translated = ShaderRecompiler::TranslateProgram(params.code, options);
			}
			if (translated->skip_dispatch) {
				entry = programs.try_emplace(lookup_key, ShaderRecompiler::IR::ResourcePlan {}).first;
				entry->second.skip_dispatch = true;
				return {};
			}
			entry = programs.try_emplace(lookup_key,
			    ShaderRecompiler::IR::ExtractResourcePlan(translated->program)).first;
			EXIT_IF(!ShaderRecompiler::IR::MaterializeResources(
			    entry->second.resource_plan, runtime, entry->second.resources,
			    entry->second.specialization, &entry->second.memo));
		}
		auto& source = entry->second;

		// Then compile the permutation for this specialization and push data position.
		std::optional<CompiledModule> compiled;
		const auto queued = std::ranges::find_if(source.pending, [&](const PendingPermutation& p) {
			return p.push_data_cursor == push_data_cursor && p.specialization == source.specialization;
		});
		if (queued != source.pending.end()) {
			if (background && !IsReady(queued->compiled)) {
				return defer();
			}
			compiled = TakeResult(queued->compiled);
			source.pending.erase(queued);
		} else {
			if (PermutationLogEnabled() && !translated) [[unlikely]] {
				LogNewPermutation(stage, params.hash, entry, push_data_cursor);
			}
			if (background) {
				std::promise<CompiledModule> promise;
				source.pending.push_back({.specialization   = source.specialization,
				                          .push_data_cursor = push_data_cursor,
				                          .compiled         = promise.get_future()});
				PostJob(
				    [device = device, input = CopyTranslationInput(stage, params, input_info),
				     translated = std::move(translated), specialization = source.specialization,
				     push_data_cursor, promise = std::move(promise)]() mutable {
					    if (!translated) {
						    translated = ShaderRecompiler::TranslateProgram(input->code, input->options);
					    }
					    promise.set_value(CompileModule(device, input->code, input->options,
					                                    std::move(*translated), specialization,
					                                    push_data_cursor));
				    },
				    urgent);
				return defer();
			}
		}
		if (compiled) {
			source.permutations.push_back(
			    MakePermutation(options, source.specialization, std::move(*compiled)));
		} else {
			if (!translated) {
				translated = ShaderRecompiler::TranslateProgram(params.code, options);
			}
			source.permutations.push_back(CompilePermutation(
			    params, options, std::move(*translated), source.specialization, push_data_cursor));
		}
		source.last_permutation = static_cast<uint32_t>(source.permutations.size() - 1u);
		source.last_push_cursor = push_data_cursor;
		const auto& permutation = source.permutations.back();
		input_info.stage = {.program = &permutation.program, .resources = &source.resources};
		if (recording.IsOpen()) {
			(void)recording.Append(ShaderPrecompile::Capture(
			    params, options, permutation.specialization, push_data_cursor, input_info));
		}
		permutation.program.bindings.AdvancePushData(push_data_cursor);

		std::array<size_t, static_cast<size_t>(ShaderType::TessellationEvaluation) + 1> counts {};
		for (const auto& [key, program_source]: programs) {
			counts[static_cast<size_t>(key.stage)] += program_source.permutations.size();
		}
		// Guest geometry shaders are compiled through the host mesh stage.
		std::printf("Shaders: VS %zu | PS %zu | CS %zu | GS %zu | LS %zu | HS %zu | TES %zu\n",
		            counts[static_cast<size_t>(ShaderType::Vertex)],
		            counts[static_cast<size_t>(ShaderType::Pixel)],
		            counts[static_cast<size_t>(ShaderType::Compute)],
		            counts[static_cast<size_t>(ShaderType::Mesh)],
		            counts[static_cast<size_t>(ShaderType::Local)],
		            counts[static_cast<size_t>(ShaderType::TessellationControl)],
		            counts[static_cast<size_t>(ShaderType::TessellationEvaluation)]);
		return permutation.handle;
	}

	// KYTY_PERMUTATION_LOG: why this lookup compiles: a new shader, a new static state of a
	// known one (and which static words differ), or a new specialization of a known source.
	template <typename Iterator>
	void LogNewPermutation(ShaderType stage, uint64_t hash, Iterator entry,
	                       uint32_t push_data_cursor) {
		std::string reason;
		if (entry != programs.end() && !entry->second.permutations.empty()) {
			const auto& source = entry->second;
			const auto& last   = source.permutations.back();
			reason = fmt::format(" new specialization ({} existing):", source.permutations.size()) +
			         SpecializationDiff(last.specialization, source.specialization);
			const auto start = ShaderRecompiler::IR::PushData::StartFor(
			    push_data_cursor, last.program.bindings.ShaderDataDwords());
			if (last.program.bindings.push_data_start_dword != start) {
				reason += fmt::format(" push_start {}->{}", last.program.bindings.push_data_start_dword,
				                      start);
			}
		} else {
			const ProgramKey* other = nullptr;
			size_t            same  = 0;
			for (const auto& [key, source]: programs) {
				if (key.hash == hash && key.stage == stage && !(key == lookup_key)) {
					other = &key;
					same++;
				}
			}
			if (other == nullptr) {
				reason = " new shader";
			} else {
				reason = fmt::format(" new static state ({} other keys):", same);
				const auto& a = other->static_state;
				const auto& b = lookup_key.static_state;
				if (a.size() != b.size()) {
					reason += fmt::format(" words {}->{}", a.size(), b.size());
				}
				size_t shown = 0;
				for (size_t i = 0; i < std::min(a.size(), b.size()) && shown < 6; i++) {
					if (a[i] != b[i]) {
						reason += fmt::format(" w[{}] {:#x}->{:#x}", i, a[i], b[i]);
						shown++;
					}
				}
				if (other->user_data_count != lookup_key.user_data_count) {
					reason += fmt::format(" user_data {}->{}", other->user_data_count,
					                      lookup_key.user_data_count);
				}
			}
		}
		const auto now = std::chrono::steady_clock::now().time_since_epoch();
		std::printf("permutation: stage=%u hash=%016llx t=%.3f%s\n", static_cast<uint32_t>(stage),
		            static_cast<unsigned long long>(hash),
		            std::chrono::duration<double>(now).count(), reason.c_str());
	}

	explicit ProgramCache(vk::Device device): device(device) {
		lookup_key.static_state.reserve(MaxStaticKeyWords);
		// Debugging aids: KYTY_VERIFY_SRT=1 checks every cached resource refresh against the
		// reference SRT walker and aborts on a difference; KYTY_SRT_STATS=1 logs how often
		// refreshes reuse descriptors.
		const auto enabled = [](const char* name) {
			const char* value = std::getenv(name);
			return value != nullptr && std::strcmp(value, "1") == 0;
		};
		if (enabled("KYTY_VERIFY_SRT")) {
			ShaderRecompiler::IR::SetResourceMaterializationVerification(true);
		}
		stats_enabled = enabled("KYTY_SRT_STATS");
	}

	void CountRefresh(const SourceEntry& source) {
		constexpr uint64_t Interval = 100000;
		stats.refreshes++;
		stats.memoizable += source.resource_plan.compiled != nullptr &&
		                    source.resource_plan.compiled->memoizable;
		stats.reused += source.memo.reused;
		if (stats.refreshes % Interval == 0) {
			PipelineCacheLog("Shader resources: {} refreshes, {} memoizable, {} reused descriptors",
			                 stats.refreshes, stats.memoizable, stats.reused);
		}
	}
	~ProgramCache() {
		for (auto& [key, entry]: programs) {
			(void)key;
			for (const auto& permutation: entry.permutations) {
				device.destroyShaderModule(permutation.handle.module, nullptr);
			}
			// The worker threads have stopped: each job either finished or was dropped.
			for (auto& queued: entry.pending) {
				if (queued.compiled.valid() && IsReady(queued.compiled)) {
					if (auto compiled = TakeResult(queued.compiled)) {
						device.destroyShaderModule(compiled->module, nullptr);
					}
				}
			}
		}
	}

	std::unordered_map<ProgramKey, SourceEntry, ProgramKeyHash> programs;
	// New sources translating on worker threads.
	std::unordered_map<ProgramKey, std::future<ShaderRecompiler::TranslateResult>, ProgramKeyHash>
	    pending_sources;
	// Whether a shader (by hash) stores data; see StoresData.
	std::unordered_map<uint64_t, bool> stores_data;
	// Runs background translations; null without pipeline libraries.
	PipelineLibraryCache*                                       workers = nullptr;
	// Background translations and module compiles: running or queued, and finished so far.
	struct JobCounts {
		std::atomic<uint32_t> in_flight {0};
		std::atomic<uint64_t> finished {0};
	};
	std::shared_ptr<JobCounts> jobs = std::make_shared<JobCounts>();
	ProgramKey                                                  lookup_key;
	vk::Device                                                  device;
	uint64_t                                                    next_shader_id = 0;
	ShaderPrecompile::Journal                                   recording;
	bool                                                        stats_enabled = false;
	struct {
		uint64_t refreshes  = 0;
		uint64_t memoizable = 0;
		uint64_t reused     = 0;
	} stats;
};

PipelineCache::PipelineCache(GraphicContext& graphics)
    : m_graphics(graphics), m_program_cache(std::make_unique<ProgramCache>(graphics.device)) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	InitializeDriverCache();
	InitializeShaderPrecompile();
	// Linking libraries without fast linking costs about as much as a full compile.
	if (m_graphics.pipeline_library_enabled && m_graphics.pipeline_library_fast_linking) {
		m_libraries = std::make_unique<PipelineLibraryCache>(m_graphics, m_driver_cache);
		m_program_cache->workers = m_libraries.get();
	}
}

PipelineCache::~PipelineCache() {
	// Save also stops the pipeline-library link thread. Pipelines linked from the libraries are
	// destroyed below, and the libraries after them.
	Save();
	auto destroy = [this](const auto& pipelines) {
		for (const auto& [key, pipeline]: pipelines) {
			(void)key;
			m_graphics.device.destroyPipeline(pipeline->pipeline, nullptr);
			m_graphics.device.destroyPipelineLayout(pipeline->pipeline_layout, nullptr);
			m_graphics.device.destroyDescriptorSetLayout(pipeline->descriptor_set_layout, nullptr);
			if (pipeline->pixel_set_layout != nullptr) {
				m_graphics.device.destroyDescriptorSetLayout(pipeline->pixel_set_layout, nullptr);
			}
		}
	};
	destroy(m_graphics_pipelines);
	destroy(m_compute_pipelines);
	// Prefetched compute pipelines no dispatch took: their pipelines belong to the library cache.
	for (const auto& [id, pipeline]: m_compute_prefetched) {
		(void)id;
		DestroyPipelineObjects(m_graphics, *pipeline);
	}
	m_libraries.reset();
	if (m_driver_cache != nullptr) {
		m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
	}
}

void PipelineCache::InitializeDriverCache() {
	const auto title_id = PipelineCacheTitleId();
	if (title_id.empty()) {
		return;
	}
	if (KYTY_BUILD != KYTY_BUILD_RELEASE) {
		PipelineCacheLog("Vulkan pipeline cache: disabled (non-Release build)");
		return;
	}
	// The key hashes the recompiler and pipeline sources, including uncommitted edits, so records
	// written by different shader code never match.
	if (std::string_view(KYTY_SHADER_CACHE_KEY) == "unknown") {
		PipelineCacheLog("Vulkan pipeline cache: disabled (unknown shader source key)");
		return;
	}

	m_driver_cache_path     = std::filesystem::path("_PipelineCache") / (title_id + ".bin");
	const auto path         = Common::PathToString(m_driver_cache_path);
	const bool cache_exists = Common::File::IsFileExisting(m_driver_cache_path);
	if (cache_exists) {
		PipelineCacheLog("Vulkan pipeline cache: loading {}", path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initializing {}", path);
	}
	std::vector<uint8_t> initial_data;
	if (cache_exists) {
		Common::File file(m_driver_cache_path, Common::File::Mode::Read);
		const auto   file_size = file.IsInvalid() ? 0 : file.Size();
		const auto   signature = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
		if (file_size >= signature.size() + sizeof(uint64_t) &&
		    file_size <= std::numeric_limits<uint32_t>::max()) {
			std::string cached_signature(signature.size(), '\0');
			uint64_t    payload_hash = 0;
			initial_data.resize(file_size - signature.size() - sizeof(payload_hash));
			uint32_t signature_read = 0;
			uint32_t hash_read      = 0;
			uint32_t payload_read   = 0;
			file.Read(cached_signature.data(), static_cast<uint32_t>(cached_signature.size()),
			          &signature_read);
			file.Read(&payload_hash, sizeof(payload_hash), &hash_read);
			file.Read(initial_data.data(), static_cast<uint32_t>(initial_data.size()),
			          &payload_read);
			file.Close();
			if (signature_read != cached_signature.size() || hash_read != sizeof(payload_hash) ||
			    payload_read != initial_data.size() || cached_signature != signature ||
			    XXH3_64bits(initial_data.data(), initial_data.size()) != payload_hash) {
				initial_data.clear();
				PipelineCacheLog(
				    "Vulkan pipeline cache: invalidating {} (driver, emulator, or data mismatch)",
				    path);
			}
		} else {
			file.Close();
			PipelineCacheLog("Vulkan pipeline cache: invalidating {} (invalid file size)", path);
		}
	}

	vk::PipelineCacheCreateInfo create {};
	create.initialDataSize = initial_data.size();
	create.pInitialData    = initial_data.empty() ? nullptr : initial_data.data();
	auto result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	if (result != vk::Result::eSuccess && !initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: driver rejected {} ({}); starting empty", path,
		                 vk::to_string(result));
		initial_data.clear();
		create.initialDataSize = 0;
		create.pInitialData    = nullptr;
		result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	}
	if (result != vk::Result::eSuccess) {
		PipelineCacheLog("Vulkan pipeline cache: disabled ({})", vk::to_string(result));
		m_driver_cache = nullptr;
		return;
	}
	if (!initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: loaded {} bytes from {}", initial_data.size(),
		                 path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initialized empty");
	}
}

void PipelineCache::InitializeShaderPrecompile() {
	// Reuse the Release/source-key/driver gate: records from different recompiler or pipeline
	// sources carry a different key and are discarded.
	if (!Config::ShaderPrecompileEnabled() || m_driver_cache == nullptr ||
	    m_driver_cache_path.empty())
		return;
	auto path = m_driver_cache_path;
	path.replace_extension(".shaders");
	auto        key = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
	std::string app_version;
	(void)Loader::SystemContentParamSfoGetString("APP_VER", &app_version);
	key += app_version;
	auto records = m_program_cache->recording.Open(path, key);
	if (records.empty()) return;
	PipelineCacheLog("Shader precompile: replaying {} recorded permutations", records.size());
	m_precompile_done.store(false, std::memory_order_release);
	m_precompile_thread = std::jthread(
	    [this, records = std::move(records)]() mutable { ReplayPrecompiled(std::move(records)); });
}

void PipelineCache::WaitForPrecompile() {
	if (m_precompile_done.load(std::memory_order_acquire)) return;
	std::lock_guard lock(m_precompile_join_mutex);
	if (m_precompile_thread.joinable()) m_precompile_thread.join();
	m_precompile_done.store(true, std::memory_order_release);
}

void PipelineCache::ReplayPrecompiled(std::vector<ShaderPrecompile::PermutationRecord> records) {
	KYTY_PROFILER_THREAD("ShaderPrecompile");
	// The first draw waits for the replay, so it translates and compiles the records on several
	// threads without the cache lock, then adds them in order under it.
	struct Replayed {
		std::optional<ShaderRecompiler::IR::ResourcePlan> plan;
		ProgramCache::CompiledModule                     compiled;
		bool                                             valid = false;
	};
	const auto options_for = [](ShaderPrecompile::PermutationRecord& record, ShaderParams& params) {
		params.code            = record.code;
		params.back_code       = record.back_code;
		params.hash            = record.hash;
		params.user_data_count = record.user_data_count;
		ShaderRecompiler::CompileOptions options;
		options.stage          = record.stage;
		options.shader_hash    = record.hash;
		options.user_data      = std::span(params.user_data).first(params.user_data_count);
		options.back_code      = params.back_code;
		options.user_data_base = record.user_data_base;
		options.wave_size      = record.wave_size;
		options.dump_ir        = false;
		options.early_dump     = false;
		options.dump_label     = "ShaderPrecompile";
		std::visit(
		    [&](auto& info) {
			    using Info = std::decay_t<decltype(info)>;
			    if constexpr (std::is_same_v<Info, ShaderVertexInputInfo>)
				    options.input_info.vertex = &info;
			    else if constexpr (std::is_same_v<Info, ShaderPixelInputInfo>)
				    options.input_info.pixel = &info;
			    else
				    options.input_info.compute = &info;
		    },
		    record.info);
		return options;
	};

	std::vector<Replayed> replayed(records.size());
	std::atomic_size_t    next {0};
	const auto            work = [&] {
		for (size_t index = next.fetch_add(1); index < records.size(); index = next.fetch_add(1)) {
			auto&        record = records[index];
			ShaderParams params;
			const auto   options    = options_for(record, params);
			auto         translated = ShaderRecompiler::TranslateProgram(params.code, options);
			// Reject inconsistent metadata before ApplyResourceSpecialization's hard assertions.
			// Live guest lookups still compile normally if a record cannot be replayed.
			if (translated.skip_dispatch || !translated.program.resource_tracking_complete ||
			    translated.program.info.buffers.size() != record.specialization.buffers.size() ||
			    translated.program.info.images.size() > record.specialization.images.size()) {
				continue;
			}
			auto& result = replayed[index];
			result.plan  = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
			result.compiled =
			    ProgramCache::CompileModule(m_graphics.device, params.code, options, std::move(translated),
			                                record.specialization, record.push_data_start_dword);
			result.valid = true;
		}
	};
	const auto threads = std::clamp(std::thread::hardware_concurrency(), 2u, 8u) - 1u;
	{
		std::vector<std::jthread> helpers;
		for (uint32_t i = 1; i < threads; i++) {
			helpers.emplace_back(work);
		}
		work();
	}

	size_t              compiled = 0;
	size_t              skipped  = 0;
	Common::LockGuard   lock(m_mutex);
	for (size_t index = 0; index < records.size(); index++) {
		auto& record = records[index];
		auto& result = replayed[index];
		if (!result.valid) {
			++skipped;
			continue;
		}
		ShaderParams                    params;
		const auto                      options = options_for(record, params);
		ProgramCache::ProgramKey        key;
		key.stage           = record.stage;
		key.hash            = record.hash;
		key.user_data_count = record.user_data_count;
		key.code_size       = static_cast<uint32_t>(record.code.size());
		std::visit([&](auto& info) { BuildStageStaticKey(info, key.static_state); }, record.info);
		auto entry = m_program_cache->programs.find(key);
		if (entry != m_program_cache->programs.end() &&
		    std::ranges::any_of(entry->second.permutations, [&](const auto& candidate) {
			    const auto& layout = candidate.program.bindings;
			    return candidate.specialization == record.specialization &&
				       layout.push_data_start_dword ==
				           ShaderRecompiler::IR::PushData::StartFor(record.push_data_start_dword,
				                                                    layout.ShaderDataDwords());
		    })) {
			m_graphics.device.destroyShaderModule(result.compiled.module, nullptr);
			continue;
		}
		if (entry == m_program_cache->programs.end()) {
			entry = m_program_cache->programs.try_emplace(std::move(key), std::move(*result.plan))
			            .first;
		}
		entry->second.permutations.push_back(m_program_cache->MakePermutation(
		    options, record.specialization, std::move(result.compiled)));
		++compiled;
	}
	PipelineCacheLog("Shader precompile: replayed {} permutations on {} threads; skipped {}",
	                 compiled, threads, skipped);
}

void PipelineCache::Save() {
	// Join before acquiring the cache mutex or destroying the driver cache. This also covers
	// the WindowRun shutdown path, which calls Save before the PipelineCache destructor.
	WaitForPrecompile();
	if (m_libraries != nullptr) {
		// The link thread writes to the driver cache, which is saved and destroyed below.
		m_libraries->Stop();
	}
	Common::LockGuard lock(m_mutex);
	m_program_cache->recording.Close();
	if (m_driver_cache == nullptr) {
		return;
	}

	size_t               size = 0;
	vk::Result           result;
	std::vector<uint8_t> payload;
	for (uint32_t attempt = 0; attempt < 3; attempt++) {
		size   = 0;
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, nullptr);
		if (result != vk::Result::eSuccess || size == 0 ||
		    size > std::numeric_limits<uint32_t>::max()) {
			break;
		}
		payload.resize(size);
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, payload.data());
		if (result != vk::Result::eIncomplete) {
			break;
		}
	}
	if (result != vk::Result::eSuccess || size == 0 ||
	    size > std::numeric_limits<uint32_t>::max()) {
		PipelineCacheLog("Vulkan pipeline cache: save failed ({}, {} bytes)",
		                 vk::to_string(result), size);
		return;
	}
	payload.resize(size);
	auto       prefix       = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
	const auto payload_hash = XXH3_64bits(payload.data(), payload.size());
	prefix.append(reinterpret_cast<const char*>(&payload_hash), sizeof(payload_hash));
	if (!Common::File::CreateDirectories(m_driver_cache_path.parent_path())) {
		PipelineCacheLog("Vulkan pipeline cache: failed to create cache directory");
		return;
	}
	auto temp_path = m_driver_cache_path;
	temp_path += ".tmp";
	Common::File file;
	uint32_t     prefix_written  = 0;
	uint32_t     payload_written = 0;
	if (file.Create(temp_path)) {
		file.Write(prefix.data(), static_cast<uint32_t>(prefix.size()), &prefix_written);
		file.Write(payload.data(), static_cast<uint32_t>(payload.size()), &payload_written);
	}
	const bool flushed = !file.IsInvalid() && file.Flush();
	file.Close();
	if (prefix_written != prefix.size() || payload_written != payload.size() || !flushed ||
	    !Common::File::RenameFile(temp_path, m_driver_cache_path)) {
		PipelineCacheLog("Vulkan pipeline cache: failed to write {}",
		                 Common::PathToString(m_driver_cache_path));
		return;
	}
	PipelineCacheLog("Vulkan pipeline cache: saved {} bytes to {}", payload.size(),
	                 Common::PathToString(m_driver_cache_path));
	m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
	m_driver_cache = nullptr;
}

PipelineCache::GraphicsPrograms PipelineCache::GetGraphicsPrograms(
    const HW::VertexShaderInfo& vertex_regs, const HW::PixelShaderInfo& pixel_regs,
    const HW::ShaderRegisters& sh, const HW::Context& context, const HW::UserConfig& user_config,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping, bool pixel_active,
    std::array<ShaderVertexInputInfo, 3>& vertex_info, ShaderPixelInputInfo& pixel_info,
    ProgramWait wait) {
	WaitForPrecompile();
	const bool tess_active = user_config.GetPrimType() == Prospero::PrimitiveType::kPatch;
	std::array<ShaderParams, 3> vertex_params;
	if (tess_active) {
		vertex_params = PrepareTessellationPrograms(vertex_regs, context, vertex_info);
	} else {
		vertex_params[0] = PrepareProgram(vertex_regs, context, user_config, vertex_info[0]);
	}
	const bool mesh_active = vertex_info[0].logical_stage == ShaderType::Mesh;
	if (mesh_active) {
		EXIT_NOT_IMPLEMENTED(!m_graphics.mesh_shader_enabled);
		auto& mesh              = vertex_info[0].mesh;
		mesh.host_subgroup_size = m_graphics.subgroup_size;
		const auto& limits      = m_graphics.mesh_shader_properties;
		const auto  logical_threads =
		    mesh.threads_num[0] * mesh.threads_num[1] * mesh.threads_num[2];
		const auto host_threads = ((logical_threads + mesh.wave_size - 1u) / mesh.wave_size) *
		                          std::min(mesh.host_subgroup_size, mesh.wave_size);
		if (host_threads > limits.maxMeshWorkGroupInvocations ||
		    host_threads > limits.maxMeshWorkGroupSize[0] ||
		    mesh.max_vertices > limits.maxMeshOutputVertices ||
		    mesh.max_primitives > limits.maxMeshOutputPrimitives ||
		    mesh.lds_size_dwords * sizeof(uint32_t) > limits.maxMeshSharedMemorySize) {
			EXIT("mesh shader exceeds host limits: threads=%u vertices=%u primitives=%u LDS=%u\n",
			     host_threads, mesh.max_vertices, mesh.max_primitives, mesh.lds_size_dwords);
		}
	}
	g_draw_phases.Mark(DrawPhaseTimer::VertexParams);
	ShaderParams pixel_params;
	if (pixel_active) {
		pixel_params = PrepareProgram(pixel_regs, sh, target_export_mapping, pixel_info);
		const auto& blend          = context.GetBlendControl(0);
		const auto  is_dual_source = [](uint8_t factor) {
			return factor >= static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Color) &&
			       factor <= static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
		};
		pixel_info.dual_source_blending =
		    blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
		    (is_dual_source(blend.color_srcblend) || is_dual_source(blend.color_destblend) ||
		     (blend.separate_alpha_blend &&
		      (is_dual_source(blend.alpha_srcblend) || is_dual_source(blend.alpha_destblend))));
		if (pixel_info.dual_source_blending) {
			// MRT1 supplies a second blend source for the same render target as MRT0.
			pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
			pixel_info.target_export_mapping[1] = pixel_info.target_export_mapping[0];
		}
	}
	if (context.GetClipControl().clip_disable) {
		const auto& viewport = context.GetScreenViewport().viewports[0];
		const auto& limits   = m_graphics.GetPhysicalDeviceProperties().limits;
		auto&       clip     = vertex_info[tess_active ? 2u : 0u].clip_space;
		clip.scale[0]        = viewport.xscale;
		clip.scale[1]        = viewport.yscale;
		clip.offset[0]       = viewport.xoffset;
		clip.offset[1]       = viewport.yoffset;
		clip.half_extent[0] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u)) * 0.5f;
		clip.half_extent[1] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u)) * 0.5f;
		clip.enabled = true;
	}
	g_draw_phases.Mark(DrawPhaseTimer::PixelParams);
	Common::LockGuard lock(m_mutex);
	uint32_t          push_data_cursor =
	    mesh_active ? ShaderRecompiler::IR::PushData::MeshDrawDwordCount : 0;
	GraphicsPrograms  result;
	if (wait == ProgramWait::Defer) {
		// A draw that stores data is never skipped (see StoresData in renderDraw.cpp), so its
		// shaders compile now.
		bool stores = pixel_active && m_program_cache->StoresData(ShaderType::Pixel, pixel_params);
		for (uint32_t i = 0; i < (tess_active ? 3u : 1u) && !stores; i++) {
			stores = m_program_cache->StoresData(vertex_info[i].logical_stage, vertex_params[i]);
		}
		if (stores) {
			wait = ProgramWait::Wait;
		}
	}
	// A pending stage stops the lookups: the push data position of the next stage depends on it.
	// A look-ahead still starts translating the later stages, which that position does not affect.
	const auto translate_rest = [&](uint32_t first_vertex) {
		if (wait == ProgramWait::Prefetch) {
			for (uint32_t i = first_vertex; i < (tess_active ? 3u : 1u); i++) {
				m_program_cache->QueueSourceTranslation(vertex_params[i], vertex_info[i],
				                                        vertex_info[i].logical_stage);
			}
		}
		return result;
	};
	if (pixel_active) {
		result.pixel =
		    m_program_cache->Get(pixel_params, pixel_info, push_data_cursor, wait, &result.pending);
		if (result.pending) {
			return translate_rest(0);
		}
	}
	g_draw_phases.Mark(DrawPhaseTimer::PixelProgram);
	for (uint32_t i = 0; i < (tess_active ? 3u : 1u); i++) {
		result.vertex[i] = m_program_cache->Get(vertex_params[i], vertex_info[i], push_data_cursor,
		                                        wait, &result.pending);
		if (result.pending) {
			return translate_rest(i + 1);
		}
	}
	g_draw_phases.Mark(DrawPhaseTimer::VertexProgram);
	return result;
}

ShaderProgram PipelineCache::GetComputeProgram(const HW::ComputeShaderInfo& regs,
                                               const HW::ShaderRegisters&   sh,
                                               ShaderComputeInputInfo&      input_info,
                                               ProgramWait wait, bool* pending) {
	WaitForPrecompile();
	input_info.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
	const auto        params      = PrepareProgram(regs, sh, input_info);
	Common::LockGuard lock(m_mutex);
	uint32_t          push_data_cursor = 0;
	return m_program_cache->Get(params, input_info, push_data_cursor, wait, pending);
}

bool PipelineStaticParameters::operator==(const PipelineStaticParameters& other) const noexcept {
	return std::memcmp(this, &other, sizeof(*this)) == 0;
}

PipelineCache::Pipeline* PipelineCache::GetGraphicsPipeline(
    std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
    std::span<const ShaderVertexInputInfo> vertex_info, CommandBuffer& command,
    const ShaderPixelInputInfo* ps_input_info, vk::PrimitiveTopology topology,
    bool primitive_restart_enable, const GraphicsPrograms& programs,
    vk::ImageAspectFlags feedback_aspects, bool may_defer) {
	const auto& vs_input_info  = vertex_info.front();
	const auto& vertex_program = programs.vertex[0];
	const auto& pixel_program  = programs.pixel;
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Gfx)", profiler::colors::DeepOrangeA200);

	EXIT_IF(colors.size() > RENDER_COLOR_ATTACHMENTS_MAX);
	EXIT_IF(!vertex_program);
	const bool ps_active = ps_input_info != nullptr;
	EXIT_IF(ps_active && !pixel_program);
	EXIT_IF(feedback_aspects && !m_graphics.attachment_feedback_loop_enabled);
	const auto color_count = static_cast<uint32_t>(colors.size());

	Common::LockGuard lock(m_mutex);
	auto&             ctx = command.GetRegisters();

	const HW::ModeControl& mc = ctx.GetModeControl();

	const auto vs_id = vertex_program.id;
	const auto ps_id = ps_active ? pixel_program.id : 0;

	GraphicsPipelineKey key {};
	for (uint32_t i = 0; i < programs.vertex.size(); i++) {
		key.vertex_shader_ids[i] = programs.vertex[i].id;
	}
	key.ps_shader_id            = ps_id;
	auto& static_params         = key.static_params;
	static_params.attachment_feedback_loop_flags = AttachmentFeedbackPipelineFlags(
	    feedback_aspects, m_graphics.attachment_feedback_loop_dynamic_enabled);
	auto& rendering             = key.rendering;
	rendering.color_count       = 0;
	uint32_t attachment_samples = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		const auto slot = colors[i].target_slot;
		EXIT_IF(slot >= RENDER_COLOR_ATTACHMENTS_MAX);
		rendering.color_count = std::max(rendering.color_count, slot + 1);
		EXIT_IF(!colors[i].image_id || colors[i].desc.view_info.format == vk::Format::eUndefined);
		static_params.color_mask[slot] = colors[i].export_mapping.ApplyMask(
		    render_target_mask_slot(ctx.GetRenderTargetMask(), colors[i].target_slot));
		rendering.color_formats[slot] = colors[i].desc.view_info.format;
		if (attachment_samples == 0) {
			attachment_samples = colors[i].desc.info.samples;
		} else if (attachment_samples != colors[i].desc.info.samples) {
			EXIT("mixed color attachment sample counts are unsupported: %u and %u\n",
			     attachment_samples, colors[i].desc.info.samples);
		}
		const auto& rt                        = ctx.GetRenderTarget(colors[i].target_slot);
		const auto& bc                        = ctx.GetBlendControl(colors[i].target_slot);
		static_params.color_srcblend[slot]       = bc.color_srcblend;
		static_params.color_comb_fcn[slot]       = bc.color_comb_fcn;
		static_params.color_destblend[slot]      = bc.color_destblend;
		static_params.alpha_srcblend[slot]       = bc.alpha_srcblend;
		static_params.alpha_comb_fcn[slot]       = bc.alpha_comb_fcn;
		static_params.alpha_destblend[slot]      = bc.alpha_destblend;
		static_params.separate_alpha_blend[slot] = bc.separate_alpha_blend;
		static_params.blend_enable[slot]         = bc.enable && !rt.info.blend_bypass;
	}
	const bool with_depth =
	    depth.desc.view_info.format != vk::Format::eUndefined && static_cast<bool>(depth.image_id);
	if (with_depth) {
		const auto aspects       = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
		rendering.depth_format   = aspects & vk::ImageAspectFlagBits::eDepth
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		rendering.stencil_format = aspects & vk::ImageAspectFlagBits::eStencil
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		if (attachment_samples == 0) {
			attachment_samples = depth.desc.info.samples;
		} else if (attachment_samples != depth.desc.info.samples) {
			EXIT("mixed color/depth sample counts are unsupported: %u and %u\n", attachment_samples,
			     depth.desc.info.samples);
		}
	}
	if (color_count == 0 && !with_depth) {
		attachment_samples = render_sample_count(ctx.GetAaConfig().msaa_num_samples);
		EXIT_IF(!static_cast<bool>(
		    m_graphics.GetPhysicalDeviceProperties().limits.framebufferNoAttachmentsSampleCounts &
		    vulkan_sample_count(attachment_samples)));
	}
	EXIT_IF(attachment_samples == 0 ||
	        vulkan_sample_count(attachment_samples) == vk::SampleCountFlagBits {});

	if (ps_active && depth.depth_test_enable && ps_input_info->ps_execute_on_noop) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 16) {
			LOGF("Pipeline: temporary: accepting EXEC_ON_NOOP with depth test enabled\n");
		}
	}

	const auto& clip_control               = ctx.GetClipControl();
	static_params.negative_one_to_one      = !clip_control.dx_clip_space;
	static_params.depth_clip_enable        = clip_control.IsZClipEnabled();
	static_params.topology                 = topology;
	static_params.primitive_restart_enable = primitive_restart_enable;
	static_params.samples                  = attachment_samples;
	static_params.sample_shading_enable =
	    ps_active && attachment_samples > 1 && ps_input_info->ps_sample_shading;
	if (static_params.sample_shading_enable && !m_graphics.sample_rate_shading_enabled) {
		EXIT("Pipeline: sample-rate shading is required but unsupported by the host\n");
	}
	static_params.depth_bounds_test_enable = depth.depth_bounds_test_enable;
	static_params.depth_min_bounds         = depth.depth_min_bounds;
	static_params.depth_max_bounds         = depth.depth_max_bounds;
	const bool rect_list = Prospero::IsRectList(command.GetUserConfig().GetPrimType());
	static_params.cull_back  = !rect_list && mc.cull_back;
	static_params.cull_front = !rect_list && mc.cull_front;
	static_params.face       = mc.face;
	static_params.provoking_vtx_last = mc.provoking_vtx_last;
	static_params.polygon_mode =
	    ResolvePolygonMode(mc, static_params.cull_front, static_params.cull_back);

	if (vs_input_info.stage.program->stage != ShaderType::Mesh) {
		EXIT_IF(vs_input_info.buffers_num < 0 ||
		        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX ||
		        vs_input_info.resources_num < 0 ||
		        vs_input_info.resources_num > ShaderVertexInputInfo::RES_MAX);
		key.vertex_input.binding_count   = static_cast<uint8_t>(vs_input_info.buffers_num);
		key.vertex_input.attribute_count = static_cast<uint8_t>(vs_input_info.resources_num);
		uint32_t attributes_num          = 0;
		for (int binding = 0; binding < vs_input_info.buffers_num; binding++) {
			const auto& buffer = vs_input_info.buffers[binding];
			EXIT_IF(buffer.attr_num < 0 || buffer.attr_num > ShaderVertexInputBuffer::ATTR_MAX);
			attributes_num += static_cast<uint32_t>(buffer.attr_num);
			EXIT_IF(attributes_num > static_cast<uint32_t>(vs_input_info.resources_num));
			key.vertex_input.bindings[binding] = {.stride   = buffer.stride,
			                                      .instance = buffer.fetch_index != 0};
			for (int attribute = 0; attribute < buffer.attr_num; attribute++) {
				const auto index = buffer.attr_indices[attribute];
				EXIT_IF(index < 0 || index >= vs_input_info.resources_num);
				key.vertex_input.attributes[index] = {
				    .offset  = buffer.attr_offsets[attribute],
				    .binding = static_cast<uint8_t>(binding),
				};
			}
		}
		EXIT_IF(attributes_num != static_cast<uint32_t>(vs_input_info.resources_num));
	}

	if (auto iter = m_graphics_pipelines.find(key); iter != m_graphics_pipelines.end()) {
		auto& found = *iter->second;
		if (found.optimize_pending) [[unlikely]] {
			InstallOptimizedPipeline(found, command);
		}
		return &found;
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(vs_input_info);
		if (ps_active) {
			ShaderDbgDumpInputInfo(*ps_input_info);
		}
		LOGF("PipelineTrace: shader modules VS=%" PRIu64 " module=%p PS=%" PRIu64 " module=%p\n",
		     vs_id, static_cast<void*>(vertex_program.module), ps_id,
		     static_cast<void*>(pixel_program.module));
	}

	if (may_defer && m_libraries != nullptr && Config::PipelineLibrariesEnabled()) {
		// Queue the shader parts no earlier pipeline built, and skip the draw until they are
		// compiled; the other two parts and the link take about a millisecond.
		bool ready = false;
		PrefetchLibraryParts(m_graphics, rendering, key.vertex_input, vertex_info, ps_input_info,
		                     programs, static_params, *m_libraries, m_driver_cache, &ready);
		if (!ready) {
			m_deferred_draws[key]++;
			return nullptr;
		}
	}
	uint32_t deferred_draws = 0;
	if (auto deferred = m_deferred_draws.find(key); deferred != m_deferred_draws.end()) {
		deferred_draws = deferred->second;
		m_deferred_draws.erase(deferred);
	}

	auto cached = std::make_unique<Pipeline>();
	LogPipelineTrace("CreatePipelineInternal begin", vs_id, ps_id);
	const auto create_start = std::chrono::steady_clock::now();
	const int library_parts =
	    CreatePipelineInternal(m_graphics, *cached, rendering, key.vertex_input, vertex_info,
	                           ps_input_info, programs, static_params, m_libraries.get(),
	                           m_driver_cache);
	m_graphics_pipelines_created++;
	if (PermutationLogEnabled()) [[unlikely]] {
		// "libs=" lists the library parts compiled: vertex input, pre-rasterization, fragment
		// shader, fragment output ("-" when all were cached); "mono" is a monolithic pipeline.
		// "pre=" lists the shader parts a prefetch had compiled.
		const auto letters = [](uint32_t bits, const char* names) {
			std::string text;
			for (uint32_t bit = 0; names[bit] != '\0'; bit++) {
				if ((bits & (1u << bit)) != 0) {
					text += names[bit];
				}
			}
			return text.empty() ? std::string("-") : text;
		};
		const auto bits = static_cast<uint32_t>(library_parts);
		const auto now = std::chrono::steady_clock::now();
		// "deferred=" counts the draws skipped while its parts compiled; "sync" marks a pipeline
		// created synchronously although asynchronous pipelines are on.
		std::string deferral;
		if (deferred_draws != 0) {
			deferral = fmt::format(" deferred={}", deferred_draws);
		} else if (!may_defer && Config::AsyncPipelinesEnabled()) {
			deferral = " sync";
		}
		std::printf("pipeline: vs=%llu ps=%llu ms=%.1f libs=%s%s%s%s t=%.3f\n",
		            static_cast<unsigned long long>(vs_id), static_cast<unsigned long long>(ps_id),
		            std::chrono::duration<double, std::milli>(now - create_start).count(),
		            library_parts < 0 ? "mono" : letters(bits & 0xfu, "VPFO").c_str(),
		            library_parts > 0 && (bits >> 5u) != 0 ? " pre=" : "",
		            library_parts > 0 && (bits >> 5u) != 0 ? letters(bits >> 5u, "PF").c_str() : "",
		            deferral.c_str(), std::chrono::duration<double>(now.time_since_epoch()).count());
	}
	LogPipelineTrace("CreatePipelineInternal done", vs_id, ps_id);

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);

	auto [iter, inserted] = m_graphics_pipelines.emplace(std::move(key), std::move(cached));
	EXIT_IF(!inserted);

	return iter->second.get();
}

uint32_t PipelineCache::PrefetchGraphicsPipeline(const HW::Context& ctx, const HW::Shader& sh,
                                                 const HW::UserConfig& user_config,
                                                 ProgramWait wait, bool* pending) {
	KYTY_PROFILER_FUNCTION();
	if (m_libraries == nullptr || !Config::PipelineLibrariesEnabled()) {
		return 0;
	}
	const auto& vs = sh.GetVs();
	const auto& ps = sh.GetPs();
	if (vs.es_regs.data_addr == 0) {
		return 0;
	}
	// Only primitive types the draw path accepts; RectList pipelines stay monolithic.
	const auto prim_type = user_config.GetPrimType();
	switch (prim_type) {
		case Prospero::PrimitiveType::kPointList:
		case Prospero::PrimitiveType::kLineList:
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriList:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip:
		case Prospero::PrimitiveType::kQuadListLegacy: break;
		case Prospero::PrimitiveType::kPatch:
			if (!Config::TessellationEnabled()) {
				return 0;
			}
			break;
		default: return 0;
	}
	// The draw path turns metadata color modes, resolves and depth/stencil copies into other
	// operations without translating their shaders, and a translation must not run that the real
	// draw never would.
	const auto color_mode = ctx.GetColorControl().mode;
	if (color_mode > 1) {
		return 0;
	}
	const auto& override    = ctx.GetDepthRenderOverride();
	const auto& depth_regs  = ctx.GetDepthRenderTarget();
	const bool  depth_copy  = override.force_z_dirty && override.force_z_valid &&
	                        depth_regs.z_info.format != Prospero::DepthFormat::kInvalid &&
	                        depth_regs.z_read_base_addr != 0 && depth_regs.z_write_base_addr != 0 &&
	                        depth_regs.z_read_base_addr != depth_regs.z_write_base_addr;
	const bool stencil_copy =
	    override.force_stencil_dirty && override.force_stencil_valid &&
	    depth_regs.stencil_info.format != Prospero::StencilFormat::kInvalid &&
	    depth_regs.stencil_read_base_addr != 0 && depth_regs.stencil_write_base_addr != 0 &&
	    depth_regs.stencil_read_base_addr != depth_regs.stencil_write_base_addr;
	if (color_mode == 0 && (depth_copy || stencil_copy)) {
		return 0;
	}

	// The draw path's decisions, made from the same registers.
	const auto& sh_regs     = ctx.GetShaderRegisters();
	const auto& db          = sh_regs.db_shader_control;
	const bool  side_effect = db.shader_kill_enable || db.shader_z_export_enable ||
	                         db.shader_mask_export_enable || db.shader_dual_export_enable ||
	                         db.shader_execute_on_noop;
	const auto target_mask = ctx.GetRenderTargetMask();
	const bool ps_active   = ps.ps_regs.data_addr != 0 &&
	                       ((target_mask & sh_regs.m_cbShaderMask) != 0 || side_effect);
	std::array<Prospero::ColorComponentMapping, RENDER_COLOR_ATTACHMENTS_MAX> export_mapping {};
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
		const auto& rt = ctx.GetRenderTarget(slot);
		if (rt.base.addr != 0 && render_target_mask_slot(target_mask, slot) != 0) {
			export_mapping[slot] = TextureGetRenderTargetFormat(rt.info.format, rt.info.channel_type,
			                                                    rt.info.channel_order)
			                           .export_mapping;
		}
	}
	std::array<ShaderVertexInputInfo, 3> vertex_info;
	ShaderPixelInputInfo                 pixel_info;
	const auto programs = GetGraphicsPrograms(vs, ps, sh_regs, ctx, user_config, export_mapping,
	                                          ps_active, vertex_info, pixel_info, wait);
	if (programs.pending) {
		*pending = true;
		return 0;
	}
	if (!programs.vertex[0] || (ps_active && !programs.pixel)) {
		return 0;
	}

	uint32_t samples = 0;
	if (ps_active) {
		for (const auto& output: pixel_info.stage.program->info.outputs) {
			if (output.kind != ShaderRecompiler::IR::StageOutputKind::Mrt ||
			    output.index >= RENDER_COLOR_ATTACHMENTS_MAX || samples != 0) {
				continue;
			}
			const auto& rt = ctx.GetRenderTarget(output.index);
			if (rt.base.addr != 0 && render_target_mask_slot(target_mask, output.index) != 0) {
				samples = render_sample_count(rt.attrib.num_fragments);
			}
		}
	}
	// As ResolveRenderDepthTarget decides whether the draw has a depth attachment.
	const auto& z           = ctx.GetDepthRenderTarget();
	const auto& rc          = ctx.GetRenderControl();
	const auto& dc          = ctx.GetDepthControl();
	const bool  has_stencil = z.stencil_info.format != Prospero::StencilFormat::kInvalid;
	const bool  depth_active =
	    dc.z_enable || dc.depth_bounds_enable || rc.depth_clear_enable || rc.copy_depth_to_color;
	const bool stencil_active =
	    has_stencil && (dc.stencil_enable || rc.stencil_clear_enable || rc.copy_stencil_to_color);
	const bool with_depth = (depth_active || stencil_active) &&
	                        (z.z_info.format != Prospero::DepthFormat::kInvalid || has_stencil);
	if (with_depth && samples == 0) {
		samples = render_sample_count(z.z_info.num_samples);
	}
	if (samples == 0 && !with_depth) {
		samples = render_sample_count(ctx.GetAaConfig().msaa_num_samples);
	}
	if (samples == 0) {
		return 0;
	}

	PipelineStaticParameters static_params {};
	const auto&              clip_control = ctx.GetClipControl();
	const auto&              mc           = ctx.GetModeControl();
	static_params.negative_one_to_one     = !clip_control.dx_clip_space;
	static_params.depth_clip_enable       = clip_control.IsZClipEnabled();
	static_params.topology                = prim_type == Prospero::PrimitiveType::kPatch
	                                            ? vk::PrimitiveTopology::ePatchList
	                                            : vk::PrimitiveTopology::eTriangleList;
	static_params.samples                 = samples;
	static_params.sample_shading_enable   = ps_active && samples > 1 && pixel_info.ps_sample_shading;
	static_params.depth_bounds_test_enable = with_depth && dc.depth_bounds_enable;
	static_params.depth_min_bounds         = ctx.GetDepthBoundsMin();
	static_params.depth_max_bounds         = ctx.GetDepthBoundsMax();
	static_params.cull_back                = mc.cull_back;
	static_params.cull_front               = mc.cull_front;
	static_params.face                     = mc.face;
	static_params.provoking_vtx_last       = mc.provoking_vtx_last;
	static_params.polygon_mode = ResolvePolygonMode(mc, static_params.cull_front, static_params.cull_back);
	if (static_params.sample_shading_enable && !m_graphics.sample_rate_shading_enabled) {
		return 0;
	}

	// The shader parts do not depend on attachment formats or vertex input, which the prediction
	// leaves empty; a depth attachment only matters as present or not.
	PipelineRenderingState rendering {};
	if (with_depth) {
		rendering.depth_format = vk::Format::eD32Sfloat;
	}
	const PipelineVertexInputState vertex_input {};
	Common::LockGuard              lock(m_mutex);
	return PrefetchLibraryParts(m_graphics, rendering, vertex_input,
	                            std::span(vertex_info).first(programs.VertexStageCount()),
	                            ps_active ? &pixel_info : nullptr, programs, static_params,
	                            *m_libraries, m_driver_cache);
}

uint32_t PipelineCache::PrefetchComputePipeline(const HW::Context& ctx, const HW::Shader& sh,
                                                uint32_t dispatch_initiator, ProgramWait wait,
                                                bool* pending) {
	KYTY_PROFILER_FUNCTION();
	if (m_libraries == nullptr || !Config::PipelineLibrariesEnabled()) {
		return 0;
	}
	const auto& cs = sh.GetCs();
	if (cs.cs_regs.data_addr == 0) {
		return 0;
	}
	// As the dispatch path prepares it (see RenderExecutor::DispatchDirect/DispatchIndirect).
	ShaderComputeInputInfo input_info {};
	input_info.dispatch_thread_dimensions =
	    (dispatch_initiator & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
	const auto program = GetComputeProgram(cs, ctx.GetShaderRegisters(), input_info, wait, pending);
	if (!program) {
		return 0;
	}
	Common::LockGuard lock(m_mutex);
	if (m_compute_pipelines.contains(program.id) || m_compute_prefetched.contains(program.id)) {
		return 0;
	}
	auto pipeline = std::make_unique<Pipeline>();
	auto create   = PrepareComputePipeline(m_graphics, *pipeline, input_info, program.module,
	                                       m_driver_cache);
	if (!m_libraries->Prefetch(ComputePrefetchKey(program.id), std::move(create))) {
		DestroyPipelineObjects(m_graphics, *pipeline);
		return 0;
	}
	m_compute_prefetched.emplace(program.id, std::move(pipeline));
	return 1;
}

void PipelineCache::LogLookahead(uint32_t draws, uint32_t parts) const {
	if (PermutationLogEnabled()) [[unlikely]] {
		std::printf("lookahead: draws=%u prefetched parts=%u t=%.3f\n", draws, parts,
		            std::chrono::duration<double>(
		                std::chrono::steady_clock::now().time_since_epoch())
		                .count());
	}
}

uint32_t PipelineCache::BackgroundShaderJobs() const noexcept {
	return m_program_cache->jobs->in_flight.load(std::memory_order_acquire);
}

uint64_t PipelineCache::BackgroundShaderJobsFinished() const noexcept {
	return m_program_cache->jobs->finished.load(std::memory_order_relaxed);
}

void PipelineCache::InstallOptimizedPipeline(Pipeline& pipeline, CommandBuffer& command) {
	const auto optimized = m_libraries->TakeOptimized(&pipeline);
	if (!optimized) {
		return;
	}
	pipeline.optimize_pending = false;
	if (*optimized == nullptr) {
		return;
	}
	// Commands recorded before this draw may still use the fast-linked pipeline.
	const auto fast_linked = pipeline.pipeline;
	pipeline.pipeline      = *optimized;
	command.GetContext().GetCommandScheduler().DeferOperation(
	    [device = m_graphics.device, fast_linked] { device.destroyPipeline(fast_linked, nullptr); });
}

PipelineCache::Pipeline&
PipelineCache::GetComputePipeline(const ShaderComputeInputInfo& input_info,
                                  const ShaderProgram&          compute_program) {
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Compute)", profiler::colors::RedA100);

	EXIT_IF(!compute_program);

	Common::LockGuard lock(m_mutex);

	if (auto iter = m_compute_pipelines.find(compute_program.id);
	    iter != m_compute_pipelines.end()) {
		return *iter->second;
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(input_info);
	}

	const auto create_start = std::chrono::steady_clock::now();
	std::unique_ptr<Pipeline> cached;
	if (const auto prefetched = m_compute_prefetched.find(compute_program.id);
	    prefetched != m_compute_prefetched.end()) {
		// A look-ahead already made its layouts and compiled (or is compiling) the pipeline.
		cached = std::move(prefetched->second);
		m_compute_prefetched.erase(prefetched);
		cached->pipeline = m_libraries->Take(ComputePrefetchKey(compute_program.id));
		if (cached->pipeline == nullptr) {
			DestroyPipelineObjects(m_graphics, *cached);
			cached = nullptr;
		}
	}
	const bool prefetched = cached != nullptr;
	if (!prefetched) {
		cached = std::make_unique<Pipeline>();
		CreatePipelineInternal(m_graphics, *cached, input_info, compute_program.module,
		                       m_driver_cache);
	}
	m_compute_pipelines_created++;
	if (PermutationLogEnabled()) [[unlikely]] {
		const auto now = std::chrono::steady_clock::now();
		std::printf("pipeline: cs=%llu ms=%.1f%s t=%.3f\n",
		            static_cast<unsigned long long>(compute_program.id),
		            std::chrono::duration<double, std::milli>(now - create_start).count(),
		            prefetched ? " pre=C" : "",
		            std::chrono::duration<double>(now.time_since_epoch()).count());
	}

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);

	auto [iter, inserted] = m_compute_pipelines.emplace(compute_program.id, std::move(cached));
	EXIT_IF(!inserted);

	return *iter->second;
}
} // namespace Libs::Graphics
