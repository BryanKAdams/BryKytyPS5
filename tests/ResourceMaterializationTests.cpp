#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <memory>

namespace {

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "ResourceMaterializationTests: failed: %s\n", text);
    std::abort();
  }
}

bool RejectSpecializationRead(void *userdata, uint64_t, std::span<uint32_t>) {
  ++*static_cast<uint32_t *>(userdata);
  return false;
}

Libs::Graphics::ShaderRecompiler::IR::Block &
AddValueBlock(Libs::Graphics::ShaderRecompiler::IR::Program &program) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto block = std::make_unique<Block>();
  auto *result = block.get();
  program.blocks.push_back(result);
  program.block_info.push_back({.id = 0});
  program.block_storage.push_back(std::move(block));
  return *result;
}

Libs::Graphics::ShaderRecompiler::IR::ResourcePlan SrtPlan(uint64_t address) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &value_block = AddValueBlock(program);

  MemoryInfo memory;
  memory.kind = ResourceKind::ScalarAddress;
  memory.planning_only = true;
  program.memory_info.push_back(memory);
  const auto low = Value(static_cast<uint32_t>(address));
  const auto high = Value(static_cast<uint32_t>(address >> 32u));
  auto &handle =
      value_block.AppendNewInst(ValueOpcode::GetAddressResource, {low, high});
  auto &raw = value_block.AppendNewInst(
      ValueOpcode::LoadAddressU32,
      {Value(&handle), Value(0u), Value(0u), Value(true)});
  raw.SetFlags(MemoryFlags{.index = 0, .pc = 0x40});
  program.srt_reads.push_back({Value(&raw), 0});

  auto &srt = value_block.AppendNewInst(ValueOpcode::GetSrtResource);
  auto &flat = value_block.AppendNewInst(ValueOpcode::ReadConst,
                                         {Value(&srt), Value(0u)});
  DescriptorSource source;
  source.dwords[0] = Value(&flat);
  source.dwords[1] = Value(0u);
  source.dword_count = 2;
  program.descriptor_sources.push_back(source);
  return ExtractResourcePlan(program);
}

Libs::Graphics::ShaderRecompiler::IR::ResourcePlan UnbasedFlatPlan() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  AddValueBlock(program);
  program.info.uses_dma = true;
  return ExtractResourcePlan(program);
}

Libs::Graphics::ShaderRecompiler::IR::ResourcePlan UserDataBufferPlan() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &value_block = AddValueBlock(program);

  auto &user_data = value_block.AppendNewInst(
      ValueOpcode::GetUserData, {Value(static_cast<ScalarReg>(0))});
  DescriptorSource source;
  source.dwords[0] = Value(&user_data);
  source.dwords[1] = Value(0u);
  source.dwords[2] = Value(0u);
  source.dwords[3] = Value(0u);
  source.dword_count = 4;
  program.descriptor_sources.push_back(source);
  program.info.buffers.push_back({.source = 0});
  return ExtractResourcePlan(program);
}

Libs::Graphics::ShaderRecompiler::IR::ResourcePlan MixedSamplerPlan() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  AddValueBlock(program);

  const auto AddSource = [&program](uint32_t dword_count, uint32_t first) {
    DescriptorSource source;
    source.dword_count = dword_count;
    source.dwords[0] = Value(first);
    for (uint32_t i = 1; i < dword_count; i++) {
      source.dwords[i] = Value(0u);
    }
    program.descriptor_sources.push_back(source);
    return static_cast<uint32_t>(program.descriptor_sources.size() - 1u);
  };

  const auto image0 = AddSource(8, 0);
  const auto image1 = AddSource(8, 0);
  const auto sampler0 = AddSource(4, 0x11111111u);
  const auto sampler1 = AddSource(4, 0x22222222u);
  program.info.images.push_back(
      {.source = image0,
       .resource_class = ImageResourceClass::Sampled,
       .numeric_class = Libs::Graphics::Prospero::TextureNumericClass::Float,
       .dimension =
           Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension::Dim2D});
  program.info.images.push_back(
      {.source = image1,
       .resource_class = ImageResourceClass::Sampled,
       .numeric_class = Libs::Graphics::Prospero::TextureNumericClass::Float,
       .dimension =
           Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension::Dim2D,
       .conversion_format =
           Libs::Graphics::Prospero::BufferFormat::k8_8_8_8UNorm});
  program.info.samplers.push_back({.source = sampler0});
  program.info.samplers.push_back({.source = sampler1});
  program.info.sampled_pairs.push_back({.image = 0, .sampler = 0});
  program.info.sampled_pairs.push_back({.image = 0, .sampler = 1});
  program.info.sampled_pairs.push_back({.image = 1, .sampler = 1});
  return ExtractResourcePlan(program);
}

void TestMappedSrtUsesDirectReaderByDefault() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  const uint32_t dword = 0x12345678;
  auto plan = SrtPlan(reinterpret_cast<uint64_t>(&dword));
  uint32_t specialization_reads = 0;
  const SrtRuntime runtime{.userdata = &specialization_reads,
                           .read_specialization_memory =
                               RejectSpecializationRead};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(plan, runtime, snapshot, specialization),
        "mapped SRT stage materialization failed");
  Check(specialization_reads == 0,
        "ordinary SRT read used the specialization reader");
  Check(snapshot.flattened_srt.size() == 1 &&
            snapshot.flattened_srt[0] == dword,
        "cache rematerialization did not use the direct reader by default");
}

void TestIntegerRuntimeValueFollowsSrtReads() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto plan = SrtPlan(0x10000);
  const auto root = plan.descriptor_sources.front().dwords[0];
  Check(ValidateRuntimeValue(plan, root, RuntimeValueType::Integer),
        "integer SRT read was rejected");

  Block values;
  auto &comparison = values.AppendNewInst(ValueOpcode::FPOrdLessThanEqual32,
                                          {Value::F32(1.f), Value::F32(0.f)});
  auto &selection = values.AppendNewInst(
      ValueOpcode::SelectU32, {Value(&comparison), Value(1u), Value(0u)});
  plan.srt_reads[0].value = Value(&selection);
  Check(ValidateRuntimeValue(plan, root),
        "ordinary SRT validation rejected a floating-point dependency");
  Check(!ValidateRuntimeValue(plan, root, RuntimeValueType::Integer),
        "integer SRT validation missed a hidden floating-point dependency");

  auto &first =
      values.AppendNewInst(ValueOpcode::ReadFirstLane, {root, Value(true)});
  Check(!ValidateRuntimeValue(plan, Value(&first), RuntimeValueType::Integer),
        "read-first-lane lost integer-only SRT validation");

  auto &active = values.AppendNewInst(ValueOpcode::ReadFirstLane,
                                      {Value(&selection), Value(&comparison)});
  Check(!ValidateRuntimeValue(plan, Value(&active), RuntimeValueType::Integer),
        "floating-point execution mask was accepted as integer-only");

  auto &lane = values.AppendNewInst(
      ValueOpcode::GetBuiltin,
      {Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId)),
       Value(0u)});
  auto &mask =
      values.AppendNewInst(ValueOpcode::INotEqual32, {Value(&lane), Value(0u)});
  selection.SetArg(0, Value(&mask));
  active.SetArg(1, Value(&mask));
  Check(ValidateRuntimeValue(plan, Value(&active), RuntimeValueType::Integer),
        "nonuniform integer execution mask was rejected");
  auto &float_value =
      values.AppendNewInst(ValueOpcode::BitCastU32F32, {Value::F32(1.f)});
  selection.SetArg(2, Value(&float_value));
  Check(!ValidateRuntimeValue(plan, Value(&active), RuntimeValueType::Integer),
        "floating-point inactive arm was accepted as integer-only");

  plan.srt_reads[0].value = Value(&first);
  Check(!ValidateRuntimeValue(plan, root, RuntimeValueType::Integer),
        "cyclic SRT read-first-lane dependency was accepted");
}

void TestUnbasedFlatCacheHitMaterializes() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto plan = UnbasedFlatPlan();
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(plan, {}, snapshot, specialization),
        "unbased FLAT stage materialization failed");
  Check(snapshot.buffers.empty() && snapshot.images.empty(),
        "unbased FLAT plan produced unexpected descriptors");
}

void TestFailedMaterializationRejectsStage() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto plan = UserDataBufferPlan();
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(!MaterializeResources(plan, {}, snapshot, specialization),
        "missing runtime user data did not reject the cached stage");
}

void TestMixedSamplerDuplicatesTheCorrectSnapshot() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto plan = MixedSamplerPlan();
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(plan, {}, snapshot, specialization),
        "mixed sampler materialization failed");
  Check(snapshot.samplers.size() == 3,
        "mixed sampler materialization appended unrelated samplers");
  Check(snapshot.samplers[2] == snapshot.samplers[1] &&
            snapshot.samplers[2] != snapshot.samplers[0],
        "point sampler variant duplicated the wrong runtime descriptor");
}

// Guest memory for the memo tests: 64 dwords at a fake GPU address.
struct TableMemory {
  static constexpr uint64_t Base = 0x10000;
  std::array<uint32_t, 64> words{};
  uint64_t base = Base;

  uint32_t strict_reads = 0;
  // Models GPU-dirty memory: strict reads fail, ordinary reads still succeed.
  bool reject_strict = false;

  static bool StrictRead(void *userdata, uint64_t address,
                         std::span<uint32_t> values) {
    auto &memory = *static_cast<TableMemory *>(userdata);
    memory.strict_reads++;
    return !memory.reject_strict && Read(userdata, address, values);
  }

  static bool Read(void *userdata, uint64_t address,
                   std::span<uint32_t> values) {
    const auto &memory = *static_cast<const TableMemory *>(userdata);
    for (size_t i = 0; i < values.size(); i++) {
      const auto at = address + i * 4u;
      if (at < memory.base || at + 4u > memory.base + sizeof(memory.words)) {
        return false;
      }
      values[i] = memory.words[(at - memory.base) / 4u];
    }
    return true;
  }
};

enum class MemoPlanExtra { None, DynamicRead, MemoryPredicate, ReadFirstLane };

// s[0:1] points at an SRT whose first four dwords are buffer 0's descriptor and
// whose fifth dword is read only by a branch condition; buffer 1 is s2 directly.
Libs::Graphics::ShaderRecompiler::IR::ResourcePlan
MemoPlan(MemoPlanExtra extra, bool condition) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &block = AddValueBlock(program);
  MemoryInfo memory;
  memory.kind = ResourceKind::ScalarAddress;
  memory.planning_only = true;
  program.memory_info.push_back(memory);
  const auto user_data = [&](uint16_t reg) {
    return Value(&block.AppendNewInst(ValueOpcode::GetUserData,
                                      {Value(static_cast<ScalarReg>(reg))}));
  };
  auto &handle = block.AppendNewInst(ValueOpcode::GetAddressResource,
                                     {user_data(0), user_data(1)});
  auto &srt = block.AppendNewInst(ValueOpcode::GetSrtResource);
  std::array<Value, 5> flat;
  for (uint32_t slot = 0; slot < flat.size(); slot++) {
    auto &raw = block.AppendNewInst(
        ValueOpcode::LoadAddressU32,
        {Value(&handle), Value(slot * 4u), Value(0u), Value(true)});
    raw.SetFlags(MemoryFlags{.index = 0, .pc = 0x40});
    program.srt_reads.push_back({Value(&raw), slot});
    flat[slot] = Value(
        &block.AppendNewInst(ValueOpcode::ReadConst, {Value(&srt), Value(slot)}));
  }
  DescriptorSource table;
  table.dword_count = 4;
  for (uint32_t dword = 0; dword < 4; dword++) {
    table.dwords[dword] = flat[dword];
  }
  program.descriptor_sources.push_back(table);
  DescriptorSource direct;
  direct.dword_count = 4;
  direct.dwords = {user_data(2), Value(0u), Value(0u), Value(0u)};
  program.descriptor_sources.push_back(direct);
  program.info.buffers.push_back({.source = 0});
  program.info.buffers.push_back({.source = 1});

  Value extra_value;
  switch (extra) {
  case MemoPlanExtra::None:
    break;
  case MemoPlanExtra::DynamicRead: {
    // A read whose offset is not immediate has no flat slot.
    auto &raw = block.AppendNewInst(
        ValueOpcode::LoadAddressU32,
        {Value(&handle), user_data(3), Value(0u), Value(true)});
    raw.SetFlags(MemoryFlags{.index = 0, .pc = 0x80});
    extra_value = Value(&raw);
    break;
  }
  case MemoPlanExtra::MemoryPredicate: {
    // Reads slots 2 and 3, which share a 64-byte block.
    auto &sum =
        block.AppendNewInst(ValueOpcode::IAdd32, {flat[3], flat[2]});
    auto &predicate = block.AppendNewInst(ValueOpcode::IEqual32,
                                          {Value(&sum), Value(0x1002u)});
    extra_value = Value(&block.AppendNewInst(
        ValueOpcode::SelectU32, {Value(&predicate), Value(7u), Value(9u)}));
    break;
  }
  case MemoPlanExtra::ReadFirstLane:
    extra_value = Value(&block.AppendNewInst(ValueOpcode::ReadFirstLane,
                                             {flat[3], Value(true)}));
    break;
  }
  if (!extra_value.IsEmpty()) {
    DescriptorSource source;
    source.dword_count = 4;
    source.dwords = {extra_value, Value(0u), Value(0u), Value(0u)};
    program.descriptor_sources.push_back(source);
    program.info.buffers.push_back({.source = 2});
  }
  auto plan = ExtractResourcePlan(program);
  if (condition) {
    // Slot 4 selects which of the two buffers is live.
    plan.control_flow.resize(3);
    auto &read_const = plan.value_storage.emplace_back(ValueOpcode::ReadConst);
    auto &srt_resource = plan.value_storage.emplace_back(ValueOpcode::GetSrtResource);
    read_const.SetArg(0, Value(&srt_resource));
    read_const.SetArg(1, Value(4u));
    plan.control_flow[0].condition = Value(&read_const);
    plan.control_flow[0].successors = {1, 2};
    plan.control_flow[1].sources = {0};
    plan.control_flow[2].sources = {1};
  }
  return plan;
}

struct MemoFixture {
  TableMemory memory;
  std::array<uint32_t, 4> user_data{};
  Libs::Graphics::ShaderRecompiler::IR::ResourceSnapshot snapshot;
  Libs::Graphics::ShaderRecompiler::IR::ResourceSpecialization specialization;
  Libs::Graphics::ShaderRecompiler::IR::MaterializationMemo memo;

  MemoFixture() {
    user_data = {static_cast<uint32_t>(TableMemory::Base), 0u, 0x2222u, 8u};
    for (uint32_t i = 0; i < memory.words.size(); i++) {
      memory.words[i] = 0x1000u + i;
    }
    memory.words[3] = 0; // Buffer 0 descriptor type 0.
  }

  bool block_reads = false;

  bool Refresh(const Libs::Graphics::ShaderRecompiler::IR::ResourcePlan &plan) {
    using namespace Libs::Graphics::ShaderRecompiler::IR;
    const SrtRuntime runtime{.user_data = user_data,
                             .read_memory = TableMemory::Read,
                             .userdata = &memory,
                             .read_specialization_memory = TableMemory::StrictRead,
                             .specialization_block_reads = block_reads};
    return MaterializeResources(plan, runtime, snapshot, specialization, &memo);
  }
};

void TestCleanBlockReads() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  // The predicate reads slots 2 and 3 strictly; the branch condition on slot 4
  // uses the direct value.
  const auto plan = MemoPlan(MemoPlanExtra::MemoryPredicate, true);
  for (const auto offset : {0u, 0x30u}) {
    MemoFixture exact;
    MemoFixture blocks;
    blocks.block_reads = true;
    for (auto *fixture : {&exact, &blocks}) {
      // With offset 0x30 the table begins inside a block, so the first block
      // read fails and falls back to exact reads.
      fixture->memory.base = TableMemory::Base + offset;
      fixture->user_data[0] = static_cast<uint32_t>(fixture->memory.base);
      fixture->memory.words[4] = 0u;
      Check(fixture->Refresh(plan), "strict-read plan failed to refresh");
    }
    Check(exact.snapshot.buffers == blocks.snapshot.buffers &&
              exact.snapshot.flattened_srt == blocks.snapshot.flattened_srt,
          "block strict reads changed the refresh");
    Check(blocks.snapshot.buffers[1].dwords[0] == 0x2222u &&
              blocks.snapshot.buffers[2].dwords[0] == 7u,
          "block strict reads misread the condition or predicate");
    Check(offset != 0u || blocks.memory.strict_reads < exact.memory.strict_reads,
          "aligned strict reads were not served from one block");
  }
}

void TestDirectConditions() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  // Every refresh reads flat SRT slots directly, so a branch condition over a
  // slot uses that value: one branch stays active even when strict reads fail.
  const auto slot_plan = MemoPlan(MemoPlanExtra::None, true);
  MemoFixture fixture;
  fixture.memory.reject_strict = true;
  fixture.memory.words[4] = 1u;
  Check(fixture.Refresh(slot_plan), "direct-condition plan failed to refresh");
  Check(CompileResourcePlan(slot_plan).direct_conditions[0] != 0u,
        "a flat-slot condition was not direct");
  Check(fixture.memory.strict_reads == 0u, "a flat-slot condition read strictly");
  Check(fixture.snapshot.buffers[0].dwords[0] != 0u &&
            fixture.snapshot.buffers[1].dwords[0] == 0u,
        "a flat-slot condition did not select its branch");

  // A condition on a read that is not a flat slot stays strict; when the
  // strict read fails, both branches stay active.
  auto dynamic_plan = MemoPlan(MemoPlanExtra::DynamicRead, true);
  dynamic_plan.control_flow[0].condition =
      dynamic_plan.descriptor_sources[2].dwords[0];
  MemoFixture dynamic;
  dynamic.memory.reject_strict = true;
  Check(dynamic.Refresh(dynamic_plan), "strict-condition plan failed to refresh");
  Check(CompileResourcePlan(dynamic_plan).direct_conditions[0] == 0u,
        "a condition on a dynamic read was direct");
  Check(dynamic.memory.strict_reads != 0u &&
            dynamic.snapshot.buffers[0].dwords[0] != 0u &&
            dynamic.snapshot.buffers[1].dwords[0] == 0x2222u,
        "a failed strict condition did not keep both branches");
}

void TestMemoTracksDescriptorInputs() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  const auto plan = MemoPlan(MemoPlanExtra::None, false);
  MemoFixture fixture;
  Check(fixture.Refresh(plan) && !fixture.memo.reused,
        "first memoized refresh reused a result");
  Check(CompileResourcePlan(plan).memoizable,
        "a plan of user data and flat SRT slots is not memoizable");
  Check(fixture.snapshot.buffers[0].dwords[1] == 0x1001u &&
            fixture.snapshot.buffers[1].dwords[0] == 0x2222u,
        "memoized refresh produced wrong descriptors");
  Check(fixture.Refresh(plan) && fixture.memo.reused,
        "unchanged inputs did not reuse the previous refresh");

  fixture.user_data[2] = 0x3333u;
  Check(fixture.Refresh(plan) && !fixture.memo.reused &&
            fixture.snapshot.buffers[1].dwords[0] == 0x3333u,
        "a user-data descriptor change was hidden by the memo");

  fixture.memory.words[1] = 0xabcdu;
  Check(fixture.Refresh(plan) && !fixture.memo.reused &&
            fixture.snapshot.buffers[0].dwords[1] == 0xabcdu,
        "an SRT table change was hidden by the memo");
  Check(fixture.snapshot.flattened_srt[1] == 0xabcdu,
        "the flat SRT buffer was not refreshed");

  // A slot read only by shader code (slot 4) refreshes the flat buffer but
  // cannot change descriptors, so the memo still applies.
  fixture.memory.words[4] = 0x4444u;
  Check(fixture.Refresh(plan) && fixture.memo.reused &&
            fixture.snapshot.flattened_srt[4] == 0x4444u,
        "a non-descriptor SRT slot was not refreshed under the memo");

  // The same table contents at another address give the same descriptors.
  fixture.memory.base = TableMemory::Base + 0x100u;
  fixture.user_data[0] = static_cast<uint32_t>(fixture.memory.base);
  Check(fixture.Refresh(plan) && fixture.memo.reused &&
            fixture.snapshot.user_data[0] == fixture.user_data[0],
        "a moved identical SRT table did not reuse descriptors");

  // A failed refresh invalidates the memo.
  fixture.user_data[0] = 0x40u;
  Check(!fixture.Refresh(plan) && !fixture.memo.valid,
        "a failed refresh left the memo valid");
  fixture.user_data[0] = static_cast<uint32_t>(fixture.memory.base);
  Check(fixture.Refresh(plan) && !fixture.memo.reused,
        "a refresh after a failure reused a stale result");
}

void TestMemoTracksActiveSources() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  const auto plan = MemoPlan(MemoPlanExtra::None, true);
  MemoFixture fixture;
  fixture.memory.words[4] = 1u;
  Check(fixture.Refresh(plan) && fixture.snapshot.buffers[0].dwords[0] != 0u &&
            fixture.snapshot.buffers[1].dwords[0] == 0u,
        "the taken branch did not select buffer 0");
  Check(fixture.Refresh(plan) && fixture.memo.reused,
        "an unchanged branch did not reuse the previous refresh");
  fixture.memory.words[4] = 0u;
  Check(fixture.Refresh(plan) && !fixture.memo.reused &&
            fixture.snapshot.buffers[0].dwords[0] == 0u &&
            fixture.snapshot.buffers[1].dwords[0] == 0x2222u,
        "a branch change was hidden by the memo");
}

void TestMemoRejectsUntrackedInputs() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  for (const auto extra :
       {MemoPlanExtra::DynamicRead, MemoPlanExtra::MemoryPredicate,
        MemoPlanExtra::ReadFirstLane}) {
    const auto plan = MemoPlan(extra, false);
    MemoFixture fixture;
    Check(fixture.Refresh(plan), "untracked-input plan failed to refresh");
    Check(!CompileResourcePlan(plan).memoizable,
          "a plan with untracked inputs was memoizable");
    Check(fixture.Refresh(plan) && !fixture.memo.reused,
          "a non-memoizable plan reused a refresh");
    const auto before = fixture.snapshot.buffers[2].dwords[0];
    // Dynamic read at s3 = 8; predicate on slots 2-3; first lane on slot 3.
    fixture.memory.words[2] ^= 0x10u;
    fixture.memory.words[3] ^= 0x20u;
    Check(fixture.Refresh(plan) &&
              fixture.snapshot.buffers[2].dwords[0] != before,
          "an untracked descriptor input change was lost");
  }
}

} // namespace

namespace Common {

int DbgExitHandler(const char *, int, std::string_view) { std::abort(); }

int DbgExitHandler(const char *, int, fmt::text_style, std::string_view) {
  std::abort();
}

int DbgExitIfHandler(const char *, const char *, int) { return 1; }

void DbgExit(int) { std::abort(); }

} // namespace Common

int main() {
  // Every refresh below is also compared with the reference SrtWalker.
  Libs::Graphics::ShaderRecompiler::IR::SetResourceMaterializationVerification(
      true);
  TestMappedSrtUsesDirectReaderByDefault();
  TestIntegerRuntimeValueFollowsSrtReads();
  TestUnbasedFlatCacheHitMaterializes();
  TestFailedMaterializationRejectsStage();
  TestMixedSamplerDuplicatesTheCorrectSnapshot();
  TestMemoTracksDescriptorInputs();
  TestMemoTracksActiveSources();
  TestMemoRejectsUntrackedInputs();
  TestCleanBlockReads();
  TestDirectConditions();
  std::puts("ResourceMaterializationTests: all cases passed");
  return 0;
}

// Keep this focused standalone target self-contained by amalgamating its small
// typed-IR implementation set.
#include "graphics/shader/recompiler/ir/Block.cpp"
#include "graphics/shader/recompiler/ir/Program.cpp"
#include "graphics/shader/recompiler/ir/Type.cpp"
#include "graphics/shader/recompiler/ir/Value.cpp"
#include "graphics/shader/recompiler/ir/opcodes/ValueOpcodes.cpp"
