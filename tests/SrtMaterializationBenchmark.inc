// Include at global scope after ShaderRecompilerComputeTests.cpp's helpers,
// before main. Requires ShaderPrecompileGpuTests.inc (shaderPrecompile.h).
//
// --srt-benchmark <journal copy>: translates every distinct shader recorded in a
// shader precompile journal (for example a copy of _PipelineCache/<title>.shaders),
// extracts its resource plan and times the per-draw cache-hit materialization.
// Never pass a live journal: this reads a file only, but keep the game's own
// journal untouched anyway.

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <set>
#include <tuple>
#include <unordered_set>
#include <xxhash.h>

namespace Libs::Graphics {
namespace {

struct SrtBenchmarkPlan {
  ShaderType stage = ShaderType::Unknown;
  uint64_t hash = 0;
  uint32_t user_data_count = 0;
  ShaderRecompiler::IR::ResourcePlan plan;
};

std::vector<ShaderPrecompile::PermutationRecord>
ReadShaderJournalCopy(const char *path) {
  std::vector<ShaderPrecompile::PermutationRecord> records;
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return records;
  }
  const std::vector<uint8_t> blob((std::istreambuf_iterator<char>(in)),
                                  std::istreambuf_iterator<char>());
  constexpr size_t MagicBytes = 8;
  if (blob.size() < MagicBytes + 8 ||
      std::memcmp(blob.data(), "KYTYSHDR", MagicBytes) != 0) {
    return records;
  }
  uint32_t key_bytes = 0;
  std::memcpy(&key_bytes, blob.data() + MagicBytes + 4, sizeof(key_bytes));
  size_t offset = MagicBytes + 8 + key_bytes;
  while (offset + 12 <= blob.size()) {
    uint32_t bytes = 0;
    uint64_t checksum = 0;
    std::memcpy(&bytes, blob.data() + offset, sizeof(bytes));
    std::memcpy(&checksum, blob.data() + offset + 4, sizeof(checksum));
    offset += 12;
    if (bytes == 0 || bytes > blob.size() - offset) {
      break;
    }
    const auto body = std::span(blob).subspan(offset, bytes);
    ShaderPrecompile::PermutationRecord record;
    if (XXH3_64bits(body.data(), body.size()) != checksum ||
        !ShaderPrecompile::Decode(body, record)) {
      break;
    }
    records.push_back(std::move(record));
    offset += bytes;
  }
  return records;
}

std::vector<SrtBenchmarkPlan>
TranslateJournalPlans(std::vector<ShaderPrecompile::PermutationRecord> &records) {
  using namespace ShaderRecompiler;
  std::vector<SrtBenchmarkPlan> plans;
  std::set<std::tuple<ShaderType, uint64_t, uint32_t, std::vector<uint32_t>>>
      seen;
  for (auto &record : records) {
    std::vector<uint32_t> static_key;
    CompileOptions options;
    options.stage = record.stage;
    options.shader_hash = record.hash;
    std::array<uint32_t, 40> zeros{};
    options.user_data = std::span(zeros).first(record.user_data_count);
    options.back_code = record.back_code;
    options.user_data_base = record.user_data_base;
    options.wave_size = record.wave_size;
    options.dump_ir = false;
    options.early_dump = false;
    options.dump_label = "SrtBenchmark";
    std::visit(
        [&](auto &info) {
          using Info = std::decay_t<decltype(info)>;
          if constexpr (std::is_same_v<Info, ShaderVertexInputInfo>) {
            options.input_info.vertex = &info;
          } else if constexpr (std::is_same_v<Info, ShaderPixelInputInfo>) {
            options.input_info.pixel = &info;
          } else {
            options.input_info.compute = &info;
          }
          BuildStageStaticKey(info, static_key);
        },
        record.info);
    if (!seen.emplace(record.stage, record.hash, record.user_data_count,
                      static_key)
             .second) {
      continue;
    }
    auto translated = TranslateProgram(record.code, options);
    if (translated.skip_dispatch ||
        !translated.program.resource_tracking_complete) {
      continue;
    }
    plans.push_back({.stage = record.stage,
                     .hash = record.hash,
                     .user_data_count = record.user_data_count,
                     .plan = IR::ExtractResourcePlan(translated.program)});
  }
  return plans;
}

struct SrtPlanShape {
  uint32_t nodes = 0;
  uint32_t selects = 0;
  uint32_t read_first_lanes = 0;
  uint32_t phis = 0;
  uint32_t dynamic_reads = 0;
  uint32_t user_data = 0;
  uint32_t descriptor_dwords = 0;
  std::map<std::string, uint32_t> opcodes;
};

SrtPlanShape AnalyzeSrtPlan(const ShaderRecompiler::IR::ResourcePlan &plan) {
  using namespace ShaderRecompiler::IR;
  SrtPlanShape shape;
  std::unordered_set<const Inst *> visited;
  std::vector<Value> pending;
  std::unordered_set<const Inst *> slot_reads;
  for (const auto &read : plan.srt_reads) {
    pending.push_back(read.value);
    if (const auto *inst = read.value.Resolve().TryInstruction()) {
      slot_reads.insert(inst);
    }
  }
  for (const auto &source : plan.descriptor_sources) {
    shape.descriptor_dwords += source.dword_count;
    for (uint32_t i = 0; i < source.dword_count; ++i) {
      pending.push_back(source.dwords[i]);
    }
    if (source.indirect_image) {
      pending.push_back(source.indirect_image->key_count);
      pending.push_back(source.indirect_image->selector_mask);
    }
  }
  for (const auto &block : plan.control_flow) {
    pending.push_back(block.condition);
  }
  for (uint32_t i = 0; i < plan.uniform_fill.fill.words; ++i) {
    pending.push_back(plan.uniform_fill.values[i]);
  }
  while (!pending.empty()) {
    const auto value = pending.back().Resolve();
    pending.pop_back();
    const auto *inst = value.TryInstruction();
    if (inst == nullptr || !visited.insert(inst).second) {
      continue;
    }
    shape.nodes++;
    const auto op = inst->GetOpcode();
    shape.opcodes[std::string(ValueOpcodeName(op))]++;
    switch (op) {
    case ValueOpcode::SelectU1:
    case ValueOpcode::SelectU32:
    case ValueOpcode::SelectF32:
      shape.selects++;
      break;
    case ValueOpcode::ReadFirstLane:
      shape.read_first_lanes++;
      break;
    case ValueOpcode::Phi:
      shape.phis++;
      break;
    case ValueOpcode::GetUserData:
      shape.user_data++;
      break;
    case ValueOpcode::LoadAddressU32:
    case ValueOpcode::ReadConstBuffer:
      if (!slot_reads.contains(inst)) {
        shape.dynamic_reads++;
      }
      break;
    default:
      break;
    }
    for (size_t i = 0; i < inst->NumArgs(); ++i) {
      pending.push_back(inst->Arg(i));
    }
  }
  return shape;
}

struct SrtFakeMemory {
  uint64_t direct_reads = 0;
  uint64_t clean_reads = 0;
  uint32_t pattern = 0;

  static uint32_t Word(uint32_t pattern, uint64_t address) {
    if (pattern == 0) {
      return 0;
    }
    // Deterministic values; half of the words are small so that pointer
    // chains and predicates take both directions.
    auto x = address * 0x9e3779b97f4a7c15ull + pattern;
    x ^= x >> 29u;
    x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 32u;
    return (x & 1u) != 0 ? static_cast<uint32_t>(x >> 8u) & 0xffu
                         : static_cast<uint32_t>(x);
  }

  static bool Direct(void *userdata, uint64_t address,
                     std::span<uint32_t> values) {
    auto &memory = *static_cast<SrtFakeMemory *>(userdata);
    memory.direct_reads++;
    for (size_t i = 0; i < values.size(); ++i) {
      values[i] = Word(memory.pattern, address + i * 4u);
    }
    return true;
  }

  static bool Clean(void *userdata, uint64_t address,
                    std::span<uint32_t> values) {
    auto &memory = *static_cast<SrtFakeMemory *>(userdata);
    memory.clean_reads++;
    for (size_t i = 0; i < values.size(); ++i) {
      values[i] = Word(memory.pattern, address + i * 4u);
    }
    return true;
  }
};

// --dump-shader <journal copy> <hash> <out dir>: writes the guest RDNA disassembly, native IR,
// SPIR-V and SPIR-V disassembly of every recorded permutation of one shader, as compiled at
// runtime (the record's own specialization and push data start).
std::optional<ShaderRecompiler::CompileResult>
CompileJournalRecord(ShaderPrecompile::PermutationRecord &record,
                     const char *label, bool dump_ir = true) {
  using namespace ShaderRecompiler;
  CompileOptions options;
  options.stage = record.stage;
  options.shader_hash = record.hash;
  std::array<uint32_t, 40> zeros{};
  options.user_data = std::span(zeros).first(record.user_data_count);
  options.back_code = record.back_code;
  options.user_data_base = record.user_data_base;
  options.wave_size = record.wave_size;
  options.dump_ir = dump_ir;
  options.early_dump = false;
  options.dump_label = label;
  std::visit(
      [&](auto &info) {
        using Info = std::decay_t<decltype(info)>;
        if constexpr (std::is_same_v<Info, ShaderVertexInputInfo>) {
          options.input_info.vertex = &info;
        } else if constexpr (std::is_same_v<Info, ShaderPixelInputInfo>) {
          options.input_info.pixel = &info;
        } else {
          options.input_info.compute = &info;
        }
      },
      record.info);
  auto translated = TranslateProgram(record.code, options);
  if (translated.skip_dispatch) {
    return std::nullopt;
  }
  return CompileProgram(std::move(translated), options, record.specialization,
                        record.push_data_start_dword);
}

void DumpJournalShader(const char *journal_path, const char *hash_text,
                       const char *out_dir) {
  constexpr const char *name = "DumpShader";
  const auto hash = std::strtoull(hash_text, nullptr, 16);
  auto records = ReadShaderJournalCopy(journal_path);
  Require(name, "journal", !records.empty(),
          std::string("no records read from ") + journal_path);
  std::filesystem::create_directories(out_dir);
  uint32_t written = 0;
  for (auto &record : records) {
    if (record.hash != hash) {
      continue;
    }
    auto compiled = CompileJournalRecord(record, name);
    if (!compiled) {
      std::printf("[host]    %s: permutation %u skips dispatch\n", name, written);
      continue;
    }
    const auto &result = *compiled;
    std::string disassembly;
    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
    (void)tools.Disassemble(result.spirv, &disassembly,
                            SPV_BINARY_TO_TEXT_OPTION_NO_HEADER |
                                SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES |
                                SPV_BINARY_TO_TEXT_OPTION_INDENT);
    const auto base = std::filesystem::path(out_dir) /
                      fmt::format("{:016x}_{}", record.hash, written);
    const auto write = [&](const char *suffix, const void *data, size_t size) {
      auto path = base;
      path += suffix;
      std::ofstream out(path, std::ios::binary);
      out.write(static_cast<const char *>(data),
                static_cast<std::streamsize>(size));
    };
    write(".rdna2", result.decoded_dump.data(), result.decoded_dump.size());
    write(".ir", result.ir_dump.data(), result.ir_dump.size());
    write(".spv", result.spirv.data(), result.spirv.size() * sizeof(uint32_t));
    write(".spvasm", disassembly.data(), disassembly.size());
    std::printf("[host]    %s: permutation %u: %zu code dwords, %zu SPIR-V words, "
                "push start %u -> %s\n",
                name, written, record.code.size(), result.spirv.size(),
                record.push_data_start_dword, base.string().c_str());
    written++;
  }
  Require(name, "shader", written != 0, "no record has that hash");
}

double MedianOf(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values.empty() ? 0.0 : values[values.size() / 2];
}

// --spirv-digest <journal copy>: recompiles every recorded permutation as the
// boot-time replay does, validates its SPIR-V, and prints one line per
// permutation (shader hash, SPIR-V digest, words, recompile time, INVALID when
// validation fails), then the totals. Diff the lines of two builds to prove a
// recompiler change leaves the SPIR-V byte-identical.
void PrintSpirvDigests(const char *journal_path) {
  constexpr const char *name = "SpirvDigest";
  auto records = ReadShaderJournalCopy(journal_path);
  Require(name, "journal", !records.empty(),
          std::string("no records read from ") + journal_path);
  spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_2);
  std::string messages;
  tools.SetMessageConsumer([&messages](spv_message_level_t, const char *,
                                       const spv_position_t &,
                                       const char *message) {
    messages += message;
    messages += '\n';
  });
  double total_ms = 0.0;
  double slowest_ms = 0.0;
  uint64_t slowest_hash = 0;
  size_t total_words = 0;
  size_t skipped = 0;
  size_t invalid = 0;
  for (auto &record : records) {
    const auto start = std::chrono::steady_clock::now();
    const auto compiled = CompileJournalRecord(record, name, false);
    const auto ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - start)
                        .count();
    total_ms += ms;
    if (ms > slowest_ms) {
      slowest_ms = ms;
      slowest_hash = record.hash;
    }
    if (!compiled) {
      skipped++;
      std::printf("digest %016llx skip\n",
                  static_cast<unsigned long long>(record.hash));
      continue;
    }
    const auto &spirv = compiled->spirv;
    total_words += spirv.size();
    messages.clear();
    const bool valid = tools.Validate(spirv);
    invalid += valid ? 0u : 1u;
    std::printf("digest %016llx %016llx %7zu %9.2f ms%s\n",
                static_cast<unsigned long long>(record.hash),
                static_cast<unsigned long long>(
                    XXH3_64bits(spirv.data(), spirv.size() * sizeof(uint32_t))),
                spirv.size(), ms, valid ? "" : " INVALID");
    if (!valid) {
      std::printf("%s", messages.c_str());
    }
  }
  std::printf("[host]    %s: %zu permutations (%zu skipped, %zu invalid), %zu SPIR-V "
              "words, recompile %.1f ms total, slowest %016llx %.1f ms\n",
              name, records.size(), skipped, invalid, total_words, total_ms,
              static_cast<unsigned long long>(slowest_hash), slowest_ms);
}

// --journal-opcodes <journal copy> <OPCODE>: lists the recorded shaders whose guest
// code uses an opcode (as the decoder names it, for example V_READFIRSTLANE_B32), with
// their stage, code size and use count.
void ListJournalOpcodes(const char *journal_path, const char *opcode) {
  constexpr const char *name = "JournalOpcodes";
  auto records = ReadShaderJournalCopy(journal_path);
  Require(name, "journal", !records.empty(),
          std::string("no records read from ") + journal_path);
  const auto needle = std::string(": ") + opcode + " ";
  std::set<uint64_t> listed;
  for (const auto &record : records) {
    // Fused and hull-front programs need their own decode path; skip them.
    if (!listed.insert(record.hash).second || !record.back_code.empty() ||
        record.stage == ShaderType::Local) {
      continue;
    }
    ShaderRecompiler::Decoder::Program decoded;
    ShaderRecompiler::Decoder::DecodeProgram(record.code, decoded);
    const auto text = ShaderRecompiler::Decoder::ProgramToString(decoded);
    size_t count = 0;
    for (auto at = text.find(needle); at != std::string::npos; at = text.find(needle, at + 1)) {
      count++;
    }
    if (count != 0) {
      std::printf("opcode %016llx stage=%u code=%zu uses=%zu\n",
                  static_cast<unsigned long long>(record.hash),
                  static_cast<unsigned>(record.stage), record.code.size(), count);
    }
  }
}

// --pipeline-compile-time <journal copy> <hash> <rounds> [spv files...]: times
// vkCreateComputePipelines for one recorded compute shader. Without files it
// times the SPIR-V this build produces; with files (for example --dump-shader
// output from two builds) it times each, interleaved round by round, against
// the layout this build derives. Each compile patches the SPIR-V generator
// word with a fresh value, so neither the driver's own shader cache nor an
// earlier round can serve it.
void TimePipelineCompile(VulkanHarness &vulkan, const char *journal_path,
                         const char *hash_text, uint32_t rounds,
                         std::span<char *const> spv_paths) {
  constexpr const char *name = "PipelineCompileTime";
  const auto hash = std::strtoull(hash_text, nullptr, 16);
  auto records = ReadShaderJournalCopy(journal_path);
  auto record = std::find_if(records.begin(), records.end(),
                             [&](const auto &item) { return item.hash == hash; });
  Require(name, "journal", record != records.end(), "no record has that hash");
  Require(name, "stage", record->stage == ShaderType::Compute,
          "only compute shaders are timed");
  std::vector<double> recompile_times;
  std::optional<ShaderRecompiler::CompileResult> compiled;
  for (uint32_t pass = 0; pass < 5; pass++) {
    const auto start = std::chrono::steady_clock::now();
    compiled = CompileJournalRecord(*record, name, false);
    recompile_times.push_back(std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - start)
                                  .count());
  }
  Require(name, "compile", compiled.has_value(), "permutation skips dispatch");
  std::printf("[host]    %s: recompiler median %.2f ms (%zu SPIR-V words)\n", name,
              MedianOf(recompile_times), compiled->spirv.size());

  // A "noopt:" prefix times that file with VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT.
  constexpr std::string_view no_optimization = "noopt:";
  std::vector<std::pair<std::string, std::vector<uint32_t>>> inputs;
  if (spv_paths.empty()) {
    inputs.emplace_back("this build", compiled->spirv);
  }
  for (const char *label : spv_paths) {
    const auto path = std::string_view(label).starts_with(no_optimization)
                          ? label + no_optimization.size()
                          : label;
    std::ifstream in(path, std::ios::binary);
    const std::vector<char> bytes((std::istreambuf_iterator<char>(in)),
                                  std::istreambuf_iterator<char>());
    Require(name, "input", !bytes.empty() && bytes.size() % 4 == 0,
            std::string("unreadable SPIR-V ") + path);
    std::vector<uint32_t> words(bytes.size() / 4);
    std::memcpy(words.data(), bytes.data(), bytes.size());
    inputs.emplace_back(label, std::move(words));
  }

  const auto device = vulkan.Device();
  std::vector<vk::DescriptorSetLayoutBinding> layout_bindings;
  for (const auto &binding : compiled->program.bindings.descriptors) {
    const auto count = NativeDescriptorCount(binding);
    if (count != 0) {
      layout_bindings.push_back(
          {ShaderRecompiler::IR::NativeBinding(ShaderType::Compute, binding.kind),
           NativeDescriptorType(binding.kind), count,
           vk::ShaderStageFlagBits::eCompute});
    }
  }
  vk::DescriptorSetLayoutCreateInfo layout_info{};
  layout_info.bindingCount = static_cast<uint32_t>(layout_bindings.size());
  layout_info.pBindings = layout_bindings.empty() ? nullptr : layout_bindings.data();
  vk::DescriptorSetLayout descriptor_layout = nullptr;
  RequireVk(name, "layout",
            device.createDescriptorSetLayout(&layout_info, nullptr, &descriptor_layout),
            "vkCreateDescriptorSetLayout");
  const vk::PushConstantRange push_range{vk::ShaderStageFlagBits::eCompute, 0,
                                         ShaderRecompiler::IR::NativePushConstantSize};
  vk::PipelineLayoutCreateInfo pipeline_layout_info{};
  pipeline_layout_info.setLayoutCount = 1;
  pipeline_layout_info.pSetLayouts = &descriptor_layout;
  pipeline_layout_info.pushConstantRangeCount = 1;
  pipeline_layout_info.pPushConstantRanges = &push_range;
  vk::PipelineLayout pipeline_layout = nullptr;
  RequireVk(name, "layout",
            device.createPipelineLayout(&pipeline_layout_info, nullptr, &pipeline_layout),
            "vkCreatePipelineLayout");

  // Distinct per run, so a later run cannot hit this run's cache entries.
  auto generator = static_cast<uint32_t>(
      std::chrono::steady_clock::now().time_since_epoch().count());
  const auto compile_ms = [&](const std::string &label, std::vector<uint32_t> &words,
                              bool fresh) {
    if (fresh) {
      words[2] = ++generator;
    }
    vk::ShaderModuleCreateInfo module_info{};
    module_info.codeSize = words.size() * sizeof(uint32_t);
    module_info.pCode = words.data();
    vk::ShaderModule module = nullptr;
    RequireVk(name, "module", device.createShaderModule(&module_info, nullptr, &module),
              "vkCreateShaderModule");
    vk::ComputePipelineCreateInfo pipeline_info{};
    pipeline_info.stage.stage = vk::ShaderStageFlagBits::eCompute;
    pipeline_info.stage.module = module;
    pipeline_info.stage.pName = "main";
    pipeline_info.layout = pipeline_layout;
    // A "noopt:" input, or KYTY_TEST_DISABLE_OPT=1 for every input, times the driver's
    // unoptimized compile instead.
    const char *disable_opt = std::getenv("KYTY_TEST_DISABLE_OPT");
    if (label.starts_with(no_optimization) ||
        (disable_opt != nullptr && disable_opt[0] == '1')) {
      pipeline_info.flags = vk::PipelineCreateFlagBits::eDisableOptimization;
    }
    vk::Pipeline pipeline = nullptr;
    const auto start = std::chrono::steady_clock::now();
    RequireVk(name, "pipeline",
              device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &pipeline),
              "vkCreateComputePipelines");
    const auto end = std::chrono::steady_clock::now();
    device.destroyPipeline(pipeline, nullptr);
    device.destroyShaderModule(module, nullptr);
    return std::chrono::duration<double, std::milli>(end - start).count();
  };

  std::vector<std::vector<double>> times(inputs.size());
  for (uint32_t round = 0; round <= rounds; round++) {
    for (size_t i = 0; i < inputs.size(); i++) {
      // Alternate the order so drift and warm-up do not favour one input.
      const auto index = (round % 2 == 0) ? i : inputs.size() - 1 - i;
      const auto ms = compile_ms(inputs[index].first, inputs[index].second, true);
      if (round != 0) {
        times[index].push_back(ms);
      }
    }
  }
  // With VK_KHR_pipeline_executable_properties, print the driver's statistics for each
  // input (VGPRs, SGPRs, instruction count, ...), and with KYTY_PIPELINE_ISA=<dir> write its
  // internal representations (the ISA) there.
  const auto *isa_dir = std::getenv("KYTY_PIPELINE_ISA");
  const auto capture = [&](const std::string &label, std::vector<uint32_t> &words) {
    words[2] = ++generator;
    vk::ShaderModuleCreateInfo module_info{};
    module_info.codeSize = words.size() * sizeof(uint32_t);
    module_info.pCode = words.data();
    vk::ShaderModule module = nullptr;
    RequireVk(name, "module", device.createShaderModule(&module_info, nullptr, &module),
              "vkCreateShaderModule");
    vk::ComputePipelineCreateInfo pipeline_info{};
    pipeline_info.stage.stage = vk::ShaderStageFlagBits::eCompute;
    pipeline_info.stage.module = module;
    pipeline_info.stage.pName = "main";
    pipeline_info.layout = pipeline_layout;
    pipeline_info.flags = vk::PipelineCreateFlagBits::eCaptureStatisticsKHR;
    if (isa_dir != nullptr) {
      pipeline_info.flags |= vk::PipelineCreateFlagBits::eCaptureInternalRepresentationsKHR;
    }
    if (label.starts_with(no_optimization)) {
      pipeline_info.flags |= vk::PipelineCreateFlagBits::eDisableOptimization;
    }
    vk::Pipeline pipeline = nullptr;
    RequireVk(name, "pipeline",
              device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &pipeline),
              "vkCreateComputePipelines");
    vk::PipelineInfoKHR info{};
    info.pipeline = pipeline;
    uint32_t executables = 0;
    RequireVk(name, "executables",
              device.getPipelineExecutablePropertiesKHR(&info, &executables, nullptr),
              "vkGetPipelineExecutablePropertiesKHR");
    for (uint32_t index = 0; index < executables; index++) {
      vk::PipelineExecutableInfoKHR executable{};
      executable.pipeline = pipeline;
      executable.executableIndex = index;
      uint32_t count = 0;
      RequireVk(name, "statistics",
                device.getPipelineExecutableStatisticsKHR(&executable, &count, nullptr),
                "vkGetPipelineExecutableStatisticsKHR");
      std::vector<vk::PipelineExecutableStatisticKHR> statistics(count);
      RequireVk(name, "statistics",
                device.getPipelineExecutableStatisticsKHR(&executable, &count,
                                                          statistics.data()),
                "vkGetPipelineExecutableStatisticsKHR");
      std::string line;
      for (const auto &statistic : statistics) {
        line += fmt::format(" {}=", statistic.name.data());
        switch (statistic.format) {
          case vk::PipelineExecutableStatisticFormatKHR::eBool32:
            line += statistic.value.b32 ? "true" : "false";
            break;
          case vk::PipelineExecutableStatisticFormatKHR::eInt64:
            line += fmt::format("{}", statistic.value.i64);
            break;
          case vk::PipelineExecutableStatisticFormatKHR::eUint64:
            line += fmt::format("{}", statistic.value.u64);
            break;
          case vk::PipelineExecutableStatisticFormatKHR::eFloat64:
            line += fmt::format("{:.3f}", statistic.value.f64);
            break;
        }
      }
      std::printf("[host]    %s: stats %s:%s\n", name, label.c_str(), line.c_str());
      if (isa_dir == nullptr) {
        continue;
      }
      count = 0;
      RequireVk(name, "isa",
                device.getPipelineExecutableInternalRepresentationsKHR(&executable, &count,
                                                                       nullptr),
                "vkGetPipelineExecutableInternalRepresentationsKHR");
      std::vector<vk::PipelineExecutableInternalRepresentationKHR> representations(count);
      RequireVk(name, "isa",
                device.getPipelineExecutableInternalRepresentationsKHR(
                    &executable, &count, representations.data()),
                "vkGetPipelineExecutableInternalRepresentationsKHR");
      std::vector<std::vector<char>> storage(count);
      for (uint32_t item = 0; item < count; item++) {
        storage[item].resize(representations[item].dataSize);
        representations[item].pData = storage[item].data();
      }
      RequireVk(name, "isa",
                device.getPipelineExecutableInternalRepresentationsKHR(
                    &executable, &count, representations.data()),
                "vkGetPipelineExecutableInternalRepresentationsKHR");
      const auto base = std::filesystem::path(isa_dir) /
                        std::filesystem::path(label.substr(label.find(':') + 1)).stem();
      for (uint32_t item = 0; item < count; item++) {
        auto path = base;
        path += fmt::format(".{}.{}.txt", index, item);
        std::ofstream out(path, std::ios::binary);
        out << representations[item].name.data() << "\n";
        out.write(storage[item].data(),
                  static_cast<std::streamsize>(representations[item].dataSize));
      }
    }
    device.destroyPipeline(pipeline, nullptr);
    device.destroyShaderModule(module, nullptr);
  };

  for (size_t i = 0; i < inputs.size(); i++) {
    auto &words = inputs[i].second;
    const auto repeat_ms = compile_ms(inputs[i].first, words, false);
    std::printf("[host]    %s: %6zu words  median %8.2f ms  min %8.2f ms  "
                "unchanged repeat %7.2f ms  %s\n",
                name, words.size(), MedianOf(times[i]),
                *std::min_element(times[i].begin(), times[i].end()), repeat_ms,
                inputs[i].first.c_str());
    if (vulkan.ExecutableInfoSupported()) {
      capture(inputs[i].first, words);
    }
  }
  device.destroyPipelineLayout(pipeline_layout, nullptr);
  device.destroyDescriptorSetLayout(descriptor_layout, nullptr);
}

void RunSrtBenchmark(const char *journal_path, uint32_t profile_seconds = 0) {
  constexpr const char *name = "SrtBenchmark";
  using namespace ShaderRecompiler;
  auto records = ReadShaderJournalCopy(journal_path);
  Require(name, "journal", !records.empty(),
          std::string("no records read from ") + journal_path);
  // Shaders without a declared hash are hashed in full by GetShaderParams on
  // every lookup.
  size_t content_hashed = 0, content_bytes = 0, code_bytes = 0;
  for (const auto &record : records) {
    code_bytes += record.code.size() * sizeof(uint32_t);
    if (record.back_code.empty() &&
        XXH3_64bits(record.code.data(), record.code.size() * sizeof(uint32_t)) ==
            record.hash) {
      content_hashed++;
      content_bytes += record.code.size() * sizeof(uint32_t);
    }
  }
  std::printf("[host]    %s: %zu/%zu records use a content hash (%zu bytes "
              "average); all records average %zu code bytes\n",
              name, content_hashed, records.size(),
              content_hashed ? content_bytes / content_hashed : 0,
              records.empty() ? 0 : code_bytes / records.size());
  {
    // Cost of one content hash of an average shader.
    std::vector<uint32_t> code(std::max<size_t>(code_bytes / records.size() / 4, 1), 7u);
    uint64_t sink = 0;
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < 100000; ++i) {
      code[0] = static_cast<uint32_t>(i);
      sink += XXH3_64bits(code.data(), code.size() * sizeof(uint32_t));
    }
    std::printf("[host]    %s: XXH3 of %zu bytes: %.1f ns (%llu)\n", name,
                code.size() * sizeof(uint32_t),
                std::chrono::duration<double, std::nano>(
                    std::chrono::steady_clock::now() - begin)
                        .count() /
                    100000.0,
                static_cast<unsigned long long>(sink & 1u));
  }
  auto plans = TranslateJournalPlans(records);
  std::printf("[host]    %s: %zu records, %zu distinct plans\n", name,
              records.size(), plans.size());

  // Static shape and per-call read counts.
  SrtPlanShape total;
  uint32_t with_control_flow = 0, with_fill = 0, with_indirect = 0,
           with_clean_slots = 0, with_selects = 0, with_rfl = 0,
           with_dynamic = 0;
  uint64_t srt_slots = 0, buffers = 0, images = 0, samplers = 0;
  for (auto &entry : plans) {
    const auto &plan = entry.plan;
    const auto shape = AnalyzeSrtPlan(plan);
    total.nodes += shape.nodes;
    total.selects += shape.selects;
    total.read_first_lanes += shape.read_first_lanes;
    total.phis += shape.phis;
    total.dynamic_reads += shape.dynamic_reads;
    total.user_data += shape.user_data;
    total.descriptor_dwords += shape.descriptor_dwords;
    for (const auto &[op, count] : shape.opcodes) {
      total.opcodes[op] += count;
    }
    with_control_flow += !plan.control_flow.empty();
    with_fill += plan.uniform_fill.fill.words != 0;
    with_indirect += plan.requires_specialization_memory;
    with_clean_slots += std::ranges::any_of(
        plan.clean_flat_slots, [](uint8_t slot) { return slot != 0; });
    with_selects += shape.selects != 0;
    with_rfl += shape.read_first_lanes != 0;
    with_dynamic += shape.dynamic_reads != 0;
    srt_slots += plan.srt_reads.size();
    buffers += plan.info.buffers.size();
    images += plan.info.images.size();
    samplers += plan.info.samplers.size();
  }
  std::printf("[host]    %s: plans=%zu nodes=%u srt_slots=%llu buffers=%llu "
              "images=%llu samplers=%llu descriptor_dwords=%u\n",
              name, plans.size(), total.nodes,
              static_cast<unsigned long long>(srt_slots),
              static_cast<unsigned long long>(buffers),
              static_cast<unsigned long long>(images),
              static_cast<unsigned long long>(samplers),
              total.descriptor_dwords);
  std::printf("[host]    %s: plans with control_flow=%u uniform_fill=%u "
              "indirect_images=%u clean_slots=%u selects=%u "
              "read_first_lane=%u dynamic_reads=%u\n",
              name, with_control_flow, with_fill, with_indirect,
              with_clean_slots, with_selects, with_rfl, with_dynamic);
  std::printf("[host]    %s: opcode histogram:", name);
  for (const auto &[op, count] : total.opcodes) {
    std::printf(" %s=%u", op.c_str(), count);
  }
  std::printf("\n");

  uint32_t memoizable = 0, conditions = 0, direct_conditions = 0, inert = 0,
           shown = 0;
  for (const auto &entry : plans) {
    const auto &compiled = IR::CompileResourcePlan(entry.plan);
    memoizable += compiled.memoizable;
    for (uint32_t block = 0; block < compiled.conditions.size(); ++block) {
      if (compiled.conditions[block] == IR::ResourceNode::NoNode) {
        continue;
      }
      conditions++;
      direct_conditions += compiled.direct_conditions[block] != 0u;
      inert += !compiled.inert_successors.empty() &&
               compiled.inert_successors[block] != 0u;
      if (compiled.direct_conditions[block] == 0u && shown < 4 &&
          std::getenv("KYTY_SRT_BENCH_SHOW") != nullptr) {
        shown++;
        // Print the condition's node tree (depth-limited).
        std::function<void(uint32_t, int)> print = [&](uint32_t index,
                                                        int depth) {
          if (index == IR::ResourceNode::NoNode || depth > 7) {
            return;
          }
          const auto &node = compiled.nodes[index];
          std::printf("[host]      %*s#%u op=%u aux=%u imm=%llu\n", depth * 2,
                      "", index, static_cast<uint32_t>(node.op), node.aux,
                      static_cast<unsigned long long>(node.imm));
          if (node.op == IR::ResourceNode::Op::ReadConst) {
            return;
          }
          for (const auto arg : node.args) {
            print(arg, depth + 1);
          }
        };
        std::printf("[host]    %s: strict condition in 0x%016llx:\n", name,
                    static_cast<unsigned long long>(entry.hash));
        print(compiled.conditions[block], 0);
      }
    }
  }
  std::printf("[host]    %s: memoizable plans=%u/%zu; conditions=%u, direct=%u, "
              "inert=%u\n",
              name, memoizable, plans.size(), conditions, direct_conditions,
              inert);

  // Two user-data sets: a draw either repeats the previous inputs or moves to
  // the other set, which changes pointers and therefore flat SRT values.
  std::array<std::array<uint32_t, 40>, 2> user_data{};
  for (uint32_t i = 0; i < user_data[0].size(); ++i) {
    user_data[0][i] = 0x10000u * (i + 1u);
    user_data[1][i] = 0x10000u * (i + 1u) + 0x40u;
  }
  SrtFakeMemory memory;
  const auto runtime_for = [&](const SrtBenchmarkPlan &entry, uint32_t set) {
    return IR::SrtRuntime{
        .user_data = std::span(user_data[set]).first(entry.user_data_count),
        .shader_base = 0x0000000800000000ull,
        .read_memory = SrtFakeMemory::Direct,
        .userdata = &memory,
        .read_specialization_memory = SrtFakeMemory::Clean,
        // As in ProgramCache; the reference walker always reads exactly.
        .specialization_block_reads = true,
    };
  };
  const auto same = [](const IR::ResourceSnapshot &a,
                       const IR::ResourceSpecialization &sa,
                       const IR::ResourceSnapshot &b,
                       const IR::ResourceSpecialization &sb) {
    return a.buffers == b.buffers && a.images == b.images &&
           a.samplers == b.samplers && a.flattened_srt == b.flattened_srt &&
           a.user_data == b.user_data && a.uniform_fill == b.uniform_fill &&
           sa == sb;
  };

  for (const auto pattern : {0u, 1u, 2u, 3u}) {
    memory.pattern = pattern;
    // Correctness: the compiled evaluator and the memo must reproduce the
    // reference walker exactly, including which plans fail.
    std::vector<uint8_t> usable(plans.size(), 0);
    uint32_t usable_count = 0, mismatches = 0, memo_hits = 0;
    uint64_t reference_direct = 0, reference_clean = 0, compiled_direct = 0,
             compiled_clean = 0;
    for (size_t i = 0; i < plans.size(); ++i) {
      for (uint32_t set = 0; set < 2; ++set) {
        IR::ResourceSnapshot expected, actual;
        IR::ResourceSpecialization expected_spec, actual_spec;
        memory.direct_reads = memory.clean_reads = 0;
        const bool ok = IR::MaterializeResourcesReference(
            plans[i].plan, runtime_for(plans[i], set), expected,
            expected_spec);
        reference_direct += memory.direct_reads;
        reference_clean += memory.clean_reads;
        memory.direct_reads = memory.clean_reads = 0;
        const bool compiled_ok = IR::MaterializeResources(
            plans[i].plan, runtime_for(plans[i], set), actual, actual_spec);
        compiled_direct += memory.direct_reads;
        compiled_clean += memory.clean_reads;
        bool match = ok == compiled_ok &&
                     (!ok || same(expected, expected_spec, actual, actual_spec));
        // Memo: evaluate the other set, then this one twice.
        IR::ResourceSnapshot memo_snapshot;
        IR::ResourceSpecialization memo_spec;
        IR::MaterializationMemo memo;
        (void)IR::MaterializeResources(plans[i].plan,
                                       runtime_for(plans[i], set ^ 1u),
                                       memo_snapshot, memo_spec, &memo);
        for (uint32_t repeat = 0; repeat < 2; ++repeat) {
          const bool memo_ok =
              IR::MaterializeResources(plans[i].plan, runtime_for(plans[i], set),
                                       memo_snapshot, memo_spec, &memo);
          memo_hits += memo.reused;
          match = match && memo_ok == ok &&
                  (!ok || same(expected, expected_spec, memo_snapshot, memo_spec));
        }
        if (!match && mismatches++ < 8) {
          std::printf("[host]    %s: pattern %u set %u: MISMATCH for %s "
                      "hash=0x%016llx (reference %s)\n",
                      name, pattern, set,
                      plans[i].stage == ShaderType::Pixel ? "PS"
                      : plans[i].stage == ShaderType::Compute ? "CS"
                                                              : "VS",
                      static_cast<unsigned long long>(plans[i].hash),
                      ok ? "ok" : "failed");
        }
        // Timed variants use both user-data sets.
        usable[i] = set == 0 ? ok : usable[i] && ok;
        usable_count += set == 1 && usable[i];
      }
    }
    std::printf("[host]    %s: pattern %u: %u/%zu plans materialize; "
                "mismatches=%u memo hits=%u; reads/pass reference "
                "direct=%llu clean=%llu, compiled direct=%llu clean=%llu\n",
                name, pattern, usable_count, plans.size(), mismatches,
                memo_hits,
                static_cast<unsigned long long>(reference_direct / 2),
                static_cast<unsigned long long>(reference_clean / 2),
                static_cast<unsigned long long>(compiled_direct / 2),
                static_cast<unsigned long long>(compiled_clean / 2));
    Require(name, "compiled evaluator", mismatches == 0,
            "compiled materialization differs from the reference walker");

    // Timing: each pass refreshes every usable plan once, like one stage of
    // each distinct shader per frame. Batches rotate the variant order.
    enum Variant { Reference, Compiled, MemoRepeat, MemoChanging, Count };
    constexpr const char *labels[] = {"reference walker", "compiled",
                                      "compiled+memo, same inputs",
                                      "compiled+memo, changing inputs"};
    constexpr size_t batches = 48;
    constexpr size_t passes = 8;
    std::vector<IR::ResourceSnapshot> snapshots(plans.size());
    std::vector<IR::ResourceSpecialization> specializations(plans.size());
    std::vector<IR::MaterializationMemo> memos(plans.size());
    std::array<std::vector<double>, Count> batch_us;
    uint32_t pass_counter = 0;
    const auto run = [&](Variant variant) {
      const auto begin = std::chrono::steady_clock::now();
      for (size_t pass = 0; pass < passes; ++pass, ++pass_counter) {
        const uint32_t set =
            variant == MemoChanging ? (pass_counter & 1u) : 0u;
        for (size_t i = 0; i < plans.size(); ++i) {
          if (!usable[i]) {
            continue;
          }
          const auto runtime = runtime_for(plans[i], set);
          bool ok = false;
          switch (variant) {
          case Reference:
            ok = IR::MaterializeResourcesReference(
                plans[i].plan, runtime, snapshots[i], specializations[i]);
            break;
          case Compiled:
            ok = IR::MaterializeResources(plans[i].plan, runtime, snapshots[i],
                                          specializations[i]);
            break;
          default:
            ok = IR::MaterializeResources(plans[i].plan, runtime, snapshots[i],
                                          specializations[i], &memos[i]);
            break;
          }
          if (!ok) {
            Fail(name, "materialize", "a usable plan stopped materializing");
          }
        }
      }
      return std::chrono::duration<double, std::micro>(
                 std::chrono::steady_clock::now() - begin)
                 .count() /
             static_cast<double>(passes * usable_count);
    };
    if (profile_seconds != 0) {
      // Profiling aid: loop one variant so that a sampler can attach.
      if (pattern != 1u) {
        continue;
      }
      const auto end = std::chrono::steady_clock::now() +
                       std::chrono::seconds(profile_seconds);
      while (std::chrono::steady_clock::now() < end) {
        (void)run(MemoChanging);
      }
      continue;
    }
    for (size_t batch = 0; batch < batches + 2; ++batch) {
      for (int step = 0; step < Count; ++step) {
        const auto variant = static_cast<Variant>((step + batch) % Count);
        // Memo variants start from a memo that matches the previous variant's
        // snapshot only by accident; reset so every batch measures steady state.
        for (auto &memo : memos) {
          memo = {};
        }
        const auto us = run(variant);
        if (batch >= 2) {
          batch_us[variant].push_back(us);
        }
      }
    }
    for (int variant = 0; variant < Count; ++variant) {
      double mean = 0;
      for (const auto us : batch_us[variant]) {
        mean += us;
      }
      mean /= static_cast<double>(batch_us[variant].size());
      std::printf("[host]    %s: pattern %u: %-32s mean %.3f us median %.3f us "
                  "per stage\n",
                  name, pattern, labels[variant], mean,
                  MedianOf(batch_us[variant]));
    }
  }
  std::printf("[host]    %-32s ok\n", name);
}

// --journal-disasm <journal copy> <hash>...: prints the decoded guest code and
// the specialized native IR of the first recorded permutation of each hash.
void RunJournalDisasm(const char *journal_path, std::span<char *const> hashes) {
  using namespace ShaderRecompiler;
  auto records = ReadShaderJournalCopy(journal_path);
  std::set<uint64_t> wanted;
  for (const auto *text : hashes) {
    wanted.insert(std::strtoull(text, nullptr, 16));
  }
  std::set<uint64_t> printed;
  for (auto &record : records) {
    if (!wanted.contains(record.hash) || !printed.insert(record.hash).second) {
      continue;
    }
    CompileOptions options;
    options.stage = record.stage;
    options.shader_hash = record.hash;
    std::array<uint32_t, 40> zeros{};
    options.user_data = std::span(zeros).first(record.user_data_count);
    options.back_code = record.back_code;
    options.user_data_base = record.user_data_base;
    options.wave_size = record.wave_size;
    options.dump_ir = true;
    options.early_dump = false;
    options.dump_label = "JournalDisasm";
    std::visit(
        [&](auto &info) {
          using Info = std::decay_t<decltype(info)>;
          if constexpr (std::is_same_v<Info, ShaderVertexInputInfo>) {
            options.input_info.vertex = &info;
          } else if constexpr (std::is_same_v<Info, ShaderPixelInputInfo>) {
            options.input_info.pixel = &info;
          } else {
            options.input_info.compute = &info;
          }
        },
        record.info);
    std::printf("=== shader %016" PRIx64 " stage %d user_data %u base %u "
                "wave %u words %zu\n",
                record.hash, static_cast<int>(record.stage),
                record.user_data_count, record.user_data_base,
                record.wave_size, record.code.size());
    auto translated = TranslateProgram(record.code, options);
    std::printf("--- decoded\n%s", translated.decoded_dump.c_str());
    if (translated.skip_dispatch ||
        !translated.program.resource_tracking_complete ||
        translated.program.info.buffers.size() !=
            record.specialization.buffers.size() ||
        translated.program.info.images.size() >
            record.specialization.images.size()) {
      std::printf("--- (no IR: record does not specialize)\n");
      continue;
    }
    auto compiled = CompileProgram(std::move(translated), options,
                                   record.specialization,
                                   record.push_data_start_dword);
    std::printf("--- IR\n%s", compiled.ir_dump.c_str());
  }
}


// --journal-specs <journal copy>: one line per recorded permutation with its
// resource specialization (buffer stride/format/swizzle, image class, dimension
// and mip count), to see how much of a permutation the bound descriptors decide.
void PrintJournalSpecializations(const char *journal_path) {
  constexpr const char *name = "JournalSpecs";
  auto records = ReadShaderJournalCopy(journal_path);
  Require(name, "journal", !records.empty(),
          std::string("no records read from ") + journal_path);
  for (const auto &record : records) {
    std::string text = fmt::format("spec {:016x} stage={} words={} push={} buffers={} images={}",
                                   record.hash, static_cast<int>(record.stage),
                                   record.code.size(), record.push_data_start_dword,
                                   record.specialization.buffers.size(),
                                   record.specialization.images.size());
    for (const auto &buffer : record.specialization.buffers) {
      text += fmt::format(" b[{:x}/{}/{:x}]", buffer.packed_stride,
                          static_cast<int>(buffer.descriptor_format),
                          buffer.descriptor_swizzle);
    }
    for (const auto &image : record.specialization.images) {
      text += fmt::format(" i[{}/{}/{}/{}/{:x}/{}{}{}]", static_cast<int>(image.numeric_class),
                          static_cast<int>(image.dimension), image.mip_count,
                          static_cast<int>(image.conversion_format), image.shader_swizzle,
                          image.indirect_search_iterations, image.cube ? "c" : "",
                          image.fmask ? "f" : "");
    }
    std::printf("%s\n", text.c_str());
  }
}

// --spec-guess <journal copy>: compiles every recorded permutation with its real
// specialization and with a guess made without descriptors (float 2D images with
// one mip level; buffers as recorded, or with their strides dropped too), and
// prints whether each guess gives byte-identical SPIR-V.
void PrintSpecializationGuesses(const char *journal_path) {
  constexpr const char *name = "SpecGuess";
  auto records = ReadShaderJournalCopy(journal_path);
  Require(name, "journal", !records.empty(),
          std::string("no records read from ") + journal_path);
  const auto digest = [](const std::vector<uint32_t> &spirv) {
    return XXH3_64bits(spirv.data(), spirv.size() * sizeof(uint32_t));
  };
  size_t images_match = 0;
  size_t all_match = 0;
  size_t compiled_count = 0;
  for (auto &record : records) {
    const auto real = CompileJournalRecord(record, name, false);
    if (!real) {
      continue;
    }
    compiled_count++;
    const auto real_digest = digest(real->spirv);
    const auto saved = record.specialization;
    for (auto &image : record.specialization.images) {
      image = {};
      image.numeric_class = static_cast<decltype(image.numeric_class)>(1);
      image.dimension = static_cast<decltype(image.dimension)>(3);
      image.mip_count = 1;
    }
    const auto images_guess = CompileJournalRecord(record, name, false);
    const bool images_ok = images_guess && digest(images_guess->spirv) == real_digest;
    for (auto &buffer : record.specialization.buffers) {
      buffer = {};
    }
    const auto all_guess = CompileJournalRecord(record, name, false);
    const bool all_ok = all_guess && digest(all_guess->spirv) == real_digest;
    record.specialization = saved;
    images_match += images_ok ? 1u : 0u;
    all_match += all_ok ? 1u : 0u;
    std::printf("guess %016llx stage=%d words=%zu images=%s buffers=%s\n",
                static_cast<unsigned long long>(record.hash),
                static_cast<int>(record.stage), real->spirv.size(),
                images_ok ? "same" : "differs", all_ok ? "same" : "differs");
  }
  std::printf("guess summary: %zu permutations, image guess identical %zu, image and "
              "buffer guess identical %zu\n",
              compiled_count, images_match, all_match);
}

// --ps-inputs <journal copy>: each recorded pixel permutation's SPI_PS_INPUT_CNTL values, marking
// DEFAULT inputs (OFFSET bit 5 without FLAT_SHADE: a constant instead of a vertex export; with
// FLAT_SHADE the bit means raw per-vertex reads), and each vertex or mesh shader's parameter
// exports, to compare what a pixel shader reads against what its partner writes.
void PrintPixelInputs(const char *journal_path) {
  constexpr const char *name = "PixelInputs";
  auto records = ReadShaderJournalCopy(journal_path);
  Require(name, "journal", !records.empty(),
          std::string("no records read from ") + journal_path);
  size_t with_default = 0;
  size_t pixel = 0;
  std::set<std::tuple<ShaderType, uint64_t, uint32_t>> exports;
  for (auto &record : records) {
    if (std::holds_alternative<ShaderVertexInputInfo>(record.info)) {
      const auto compiled = CompileJournalRecord(record, name, false);
      uint32_t mask = 0;
      for (const auto &output : compiled ? compiled->program.info.outputs
                                         : decltype(compiled->program.info.outputs){}) {
        if (output.kind == ShaderRecompiler::IR::StageOutputKind::Parameter &&
            output.index < 32u) {
          mask |= 1u << output.index;
        }
      }
      exports.emplace(record.stage, record.hash, mask);
      continue;
    }
    const auto *info = std::get_if<ShaderPixelInputInfo>(&record.info);
    if (info == nullptr) {
      continue;
    }
    pixel++;
    std::string text = fmt::format("ps {:016x} inputs={}", record.hash, info->input_num);
    bool has_default = false;
    for (uint32_t i = 0; i < info->input_num; i++) {
      const auto value = info->interpolator_settings[i];
      text += fmt::format(" {:08x}", value);
      has_default = has_default || ((value & 0x20u) != 0 && (value & 0x400u) == 0);
    }
    with_default += has_default ? 1u : 0u;
    std::printf("%s%s%s%s%s\n", text.c_str(), has_default ? " DEFAULT" : "",
                info->ps_early_z ? " early-z" : "", info->ps_pixel_kill_enable ? " kill" : "",
                info->ps_depth_export_enable ? " z-export" : "");
  }
  for (const auto &[stage, hash, mask] : exports) {
    std::printf("%s %016llx params=%08x\n", stage == ShaderType::Mesh ? "ms" : "vs",
                static_cast<unsigned long long>(hash), mask);
  }
  std::printf("pixel permutations: %zu, with a default-value input %zu\n", pixel,
              with_default);
}
} // namespace
} // namespace Libs::Graphics
