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
#include <fstream>
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

double MedianOf(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values.empty() ? 0.0 : values[values.size() / 2];
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

  uint32_t memoizable = 0;
  for (const auto &entry : plans) {
    memoizable += IR::CompileResourcePlan(entry.plan).memoizable;
  }
  std::printf("[host]    %s: memoizable plans=%u/%zu\n", name, memoizable,
              plans.size());

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

} // namespace
} // namespace Libs::Graphics
