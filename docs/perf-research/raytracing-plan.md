# Ray tracing for Astro Bot: research plan

Status on September 25, 2026: research and design only. This branch changes no emulator behavior.
Branch `research/raytracing`, based on `93140bc8` (`perf/astro-amd-20260924`).

Target: Astro Bot (PPSA21564, US 01.018) with its ray-tracing paths on, at a playable frame rate on a
Ryzen 7 7800X3D and RX 9070 XT (AMD Windows Vulkan driver 2.0.395, reported as `26.6.4 (LLPC)`).
Today the game runs with a patch that selects the non-tiled deferred renderer and disables GI probes,
and the emulator skips every compute dispatch that contains `IMAGE_BVH_INTERSECT_RAY` (MIMG `0xE6`)
or `IMAGE_BVH64_INTERSECT_RAY` (`0xE7`).

Evidence labels used below: **verified** (read in code or data, or measured here), **reported**
(stated by a third party in a PR or issue, not reproduced here), **inferred** (reasoned, not yet
tested).

## Summary

1. **Node formats.** The PS5 GPU die is reported to be the one Mesa calls GFX1013 (Cyan Skillfish, also
   sold as the BC-250 board). Mesa gives it the same ray-tracing IP level as RDNA2, `RT_1_1`, and RADV
   runs hardware ray tracing on it with the same RTIP 1.1 node layouts it uses on RDNA2 GPUs. The
   instruction's node layout is fixed by the hardware, so Astro Bot's BVH almost certainly uses the
   documented RTIP 1.1 nodes that PR #640 implements. PR #558's "shared-exponent box node with
   implicit child pointers" has no support in the gfx10 descriptor, conflicts with the RTIP 1.x node
   types, resembles AMD's later RDNA4 quantized node, and was committed as "untested". A one-time
   capture run (Stage 1) settles this with the game's own data.
2. **Driver compile time.** The minutes-long compiles come from the recompiler's dispatcher fallback,
   not from shader size. In a standalone test on this PC, the same 128 blocks and 1,024 values
   compiled in **0.48 s** as structured code (23 VGPRs, no scratch). As a dispatcher, the compile was
   still running when a 20 s cap stopped it. The dispatcher makes every value that crosses a block
   live around one big loop. The RT kernels fall back because the structurizer cannot handle their
   loop "exit tails". Upstream `main` emits invalid SPIR-V for them, and #640 turned that into the
   dispatcher. The fix is a structurizer change (multi-exit loop normalization), not a smaller
   intersection routine. Making the dispatcher's `pc` wave-uniform did not help.
3. **#558's 3 fps.** It is mostly fixable. The branch decodes nodes with the likely-wrong format and
   reports every triangle as a miss, so rays never stop early. It fetches each node as 16 dword
   address loads through the BDA page-table path. It also adds full GPU drains (occlusion-query reuse,
   indirect arguments, CondExec), and each drain now waits behind the slow RT kernels. The reported
   profile (GPU about 50% busy, GPU thread blocked in `CommandScheduler::Wait`) fits serialized waits,
   not a saturated GPU. The fork already runs indirect dispatches on the GPU. A rough first-principles
   estimate puts correct software traversal at a few milliseconds per frame for GI-scale ray counts.
4. **Effects.** No public source documents Astro Bot's RT effects, and Digital Foundry's review does
   not mention any (checked through forum summaries, not the video itself). The patch has two
   independent switches: the tiled deferred-lighting renderer, and "GI probes and lighting shaders".
   The likely uses are probe GI and lighting in the tiled path. Each switch can be turned on
   separately, and the emulator can add a per-kernel allow list.
5. **Plan.** Seven stages. The default path stays byte-for-byte what it is now: `--ray-tracing skip`
   is the default, and every recompiler change is gated on the shader containing a BVH instruction
   until a journal diff proves it changes no other shader.

**Recommended first step:** Stage 0 plus Stage 1, the opt-in capture mode and the offline
analysis/journal tools. It needs no GPU work of its own and one short game session. That session
confirms the node format, names the kernels behind each patch switch, and yields the guest code of
all BVH kernels. With that code, Stages 2 and 3 can be developed and tested offline.

## 1. How the PS5 instruction works

The instruction tests **one node**. Traversal (stack, loop, instance transforms, closest-hit
bookkeeping) is ordinary guest shader code around it. This follows the RDNA2 ISA and GPURT
(`IntersectCommon.hlsl`, `image_bvh64_intersect_ray_base`), and matches the Astro Bot encodings
captured in `tests/ShaderRayTracingTests.inc`.

- **Operands (verified in LLVM `SIISelLowering.cpp`, `amdgcn_image_bvh_intersect_ray`):** node pointer
  (1 dword, or 2 for BVH64), ray extent (tmax), origin xyz, direction xyz, inverse direction xyz. That
  is 11 or 12 VGPRs, or 8 or 9 with A16, where direction and inverse direction are packed halves. The
  instruction writes 4 dwords to VDATA. The encoding requires R128=1, DIM=1D, DMASK=0xF, and supports
  NSA.
- **Captured Astro Bot forms (verified by decoding):** `f1989f01 00081014` is BVH32, contiguous
  address v20, result v16, T# s[32:35]. `f1989f07 00071001 3d3e3c15 38095851 00003a39` is BVH32 NSA
  with 3 extra address dwords, T# s[28:31]. #640's AMD tester logged `f1989f07 00040505 4442413d
  4543403e 00004746`. All are `0xE6` without A16. Other titles in issue #281 use `0xE7`.
- **T# (`sq_bvh_rsrc_t`, PAL `gfx10_sq_ko_reg.h`, verified):** base_address[39:0] in 256-byte units,
  15 reserved bits, box_grow_value (8 bits, ULPs), box_sort_en (bit 63), size (42 bits, node count
  minus 1 per PAL `CreateBvhSrds`), triangle_return_mode (bit 120), llc_noalloc, big_page, and type
  (bits 124-127, `0x8`). No field selects a node format.
- **Node pointer (verified, GPURT):** bits [2:0] are the node type. The rest is the byte offset / 8, so
  nodes are 64-byte aligned. Address = base*256 + ((ptr & ~7) << 3).
- **Node types (RTIP 1.x, GPURT `ShaderDefs.hlsli`):** 0-3 triangle, 4 box16 (64 B), 5 box32 (128 B),
  6 instance and 7 procedural. Types 6 and 7 are software ("user") nodes; the hardware returns all
  ones.
- **Results:** a box node returns four child pointers, misses replaced by `0xFFFFFFFF`, sorted by entry
  distance when box_sort_en is set. A triangle node returns `{t_num, t_denom, i_num, j_num}` with
  return mode 1, or `{t_num, t_denom, triangle_id, hit_status}` with mode 0. A miss sets t_num=+inf
  and t_denom=1.

Every call site's cost is therefore one node fetch (64 or 128 bytes) plus about 100-300 ALU operations,
repeated for every node the guest loop visits.

## 2. Question 1: which node formats Astro Bot uses

### Evidence

| Source | What it says | Status |
| --- | --- | --- |
| Mesa `src/amd/common/ac_gpu_info.c` (fetched September 25, 2026) | `/* GFX1013 is GFX10 plus ray tracing instructions */` `has_image_bvh_intersect_ray = gfx_level >= GFX10_3 \|\| family == CHIP_GFX1013`, and `rt_ip_version = RT_1_1` for GFX10_3 **or GFX1013** | verified |
| Phoronix on Mesa MR 33116 (Cyan Skillfish/BC-250 in RADV) | Quake II RTX runs through the RT pipeline, "3-4x faster than `RADV_PERFTEST=emulate_rt`" | reported |
| GFX1013 = the PS5 GPU die (Cyan Skillfish / BC-250) | Phoronix, BC-250 community docs | reported, widely known |
| RADV traversal (`radv_nir_rt_common.c`) | walks `radv_bvh_box32_node`, box16 and triangle nodes and passes them to `nir_bvh64_intersect_ray_amd` (the hardware instruction) | verified in ref copy |
| PAL `sq_bvh_rsrc_t` | no format-select bit; 15 + 14 reserved bits | verified |
| #558 `RayTracing.cpp` | decodes **every** node with type > 1 as a 64-byte node: 29-bit child base, 4 x 3-bit child types, 3 x 8-bit exponents at bit 56, 24 x 18-bit bounds from bit 80; treats types 2/3 as "big fp32 / big shared-exponent box"; returns a miss for types 0/1 | verified (code), commit `d995d855` is titled "(untested)" |
| GPURT gfx12 `QuantizedBVH8BoxNode` (RTIP 3.x, RDNA4) | per-axis 8-bit exponents packed in one dword, implicit child addressing, 8 children | verified in ref copy |

### Reconciliation

The hardware instruction decodes the node, so the format is a property of the silicon, not of Sony's
BVH builder. The same die running RADV's RTIP 1.1 BVHs through the real instruction shows that it
reads RDNA2's node layouts (at least the box32 and triangle nodes RADV relies on) the way RDNA2
does (inferred from the above, strong).

#558's layout has three problems:

- It uses types 2 and 3 as box formats, but in RTIP 1.x they are triangle types.
- It ignores box16/box32 entirely.
- Its ingredients (shared per-axis exponents, implicit child pointers) match RTIP 3.x's quantized
  node, not RTIP 1.x.

Nothing in the gfx10 descriptor could switch the hardware to it. The most likely explanation is that
#558's format is a guess, and that on real data it produces garbage bounds and child pointers.

Remaining uncertainty (small, but it must be checked with data):

- Sony could use reserved T# bits or undocumented node encodings that PC drivers never emit.
- RTIP 1.x triangle types 2 and 3 are not used by current GPURT (only tri0 = v0,v1,v2 and
  tri1 = v1,v3,v2), so their vertex selection is undocumented.
- The instance node (type 6) layout is defined by Sony's software, not by the hardware.

### How to confirm: opt-in capture mode (Stage 1)

`--ray-tracing capture` runs the normal translation for BVH kernels, captures data, and then still
skips the dispatch. It never creates a pipeline for these kernels, so there is no multi-minute driver
compile and no risk of losing the device. Per BVH kernel, on its first dispatch (and again after
N=600 frames, to see rebuilds):

- **Kernel record:** shader hash, guest code (`.bin`, plus `.rdna2` disassembly), `ShaderComputeInputInfo`
  (threads, LDS, scratch, wave size), user SGPRs, and dispatch dimensions, or the indirect-args
  address plus its current contents. Also every BVH instruction (pc, opcode, A16, NSA registers, T#
  SGPR base) and the CFG outcome (structured or dispatcher, and the reason from
  `LogDispatcherFallback`, `ShaderRecompiler.cpp:61`). Write the record in the existing journal
  format (`ShaderPrecompile::Capture`, `shaderPrecompile.h`) into a separate `.rt.shaders` file, so
  offline tools can recompile it with its real specialization.
- **T# values:** evaluate the descriptor's source with the SRT evaluator. The frontend records each
  BVH instruction's four T# dwords as plan sources, like #640's `bvh_sources`, and
  `MaterializeResources` evaluates them at dispatch time. Log the decoded fields, and flag nonzero
  reserved bits or a type other than `0x8`.
- **Node pool:** read `[base, base + min((size+1)*64, 256 MiB))` through
  `Memory::TryReadGpuCleanBacking` after publishing GPU-dirty pages. In capture mode only, this may
  drain the GPU. Save one file per distinct base.
- **Where:** in `PipelineCache::ProgramCache::Get` (`pipelineCache.cpp:289` and `:357`), the
  skip-entry path, when the capture mode is on. Put the capture code in a new
  `src/graphics/host_gpu/renderer/rtCapture.{h,cpp}` so the hot path gains only one branch on a
  cached bool, taken for BVH kernels.
- **Output folder:** `_TempData/rt-capture/PPSA21564/`. The files derive from a commercial game:
  keep them local, and never commit or post them (#640's AMD tester withheld dumps for the same
  reason).

An offline analyzer, `bvh_inspect` (a mode of a test executable, section 7), scores two hypotheses over
every box node reachable from observed root pointers. H1 is RTIP 1.1 (types 4/5 as box16/box32,
0/1 as triangles). H2 is #558's layout. It checks:

- child pointers are `0xFFFFFFFF` or have index <= size
- bounds are finite, with min <= max on each axis
- each child's own boxes lie inside the box its parent stores for it (within fp16 rounding and
  box_grow)
- no cycles
- triangles lie inside their parent box and are not degenerate
- the triangle_id byte fields are consistent with the node type

Accept a format when at least 99.9% of reachable nodes pass. The analyzer also histograms node
types and dumps the first 128 bytes of type-6 nodes with candidate decodings (a 3x4 float transform
and a BLAS root pointer in the pool) to pin down the instance layout. A second, GPU-side trace
(Stage 3+, emulate mode) records `{node pointer, ray, raw node dwords, emulated result}` for up to
65,536 calls into a debug buffer, to check the emulation on real rays.

## 3. Question 2: why the AMD driver takes minutes, and how to avoid it

### What was observed

- **#640 on RX 9070 XT (reported, theantipopau, driver 26.8.1 LLPC):** kernels `cs_503d7f066a496c3c`
  (610 KB, about 3 min), `cs_1f16e50eea0c89e3` (649 KB, about 6 min) and `cs_ff8ee744ffa4dcdc`
  (683 KB, still compiling after more than 6 min). Each has one `OpLoopMerge` + `OpSwitch` (the
  dispatcher fallback), about 1,050 function-scope `OpVariable`s, about 710 `OpPhi`s and about 230
  `OpFunctionCall`s (the `get_bda_pointer` helper). `spirv-val` passes.
- **#558 on the same GPU model (reported):** its RT kernels are 269k-309k words (about 1.1-1.2 MB), about
  twice #640's size, and structured. There were "no long driver compile stalls". Its comment on
  forcing the dispatcher for these kernels: "costs roughly 65x frame time on Astro Bot".
- **Why they fall back (verified in code):** in the fork (same as upstream `main`),
  `IsInnermostLoopControlConditional` (`ShaderCFG.cpp:1239`) returns true whenever one branch target
  is inside the natural-loop body and the other is not (`:1255-1259`). The branch is then emitted
  without `OpSelectionMerge`. SPIR-V allows that only when a target is the loop's merge or continue
  block. For branches to an "exit tail", a block outside the body that is not the loop merge
  (`ComputeNaturalLoops` picks the merge as the common post-dominator of all exits, `:836-871`), the
  output is invalid SPIR-V. #558's `ceac968f` notes that "Astro Bot's GI compute kernels all come out
  that way". #640 removed the shortcut, so those kernels now fail structurization and take the
  dispatcher (`EmitDispatcherFunction`, `spirvEmitterProgram.cpp:373`).

### Why the dispatcher is so expensive (measured here)

The dispatcher makes every value that crosses a block a function variable (`spirvEmitterProgram.cpp:661-714`).
Blocks are cases of one switch inside one loop. After the driver's mem2reg, every such value is
carried around that loop. It is live in every case at once, so register pressure is roughly the number
of cross-block values (about 1,050 here) instead of the guest's own at most 256 VGPRs. The register
allocator spills and scales superlinearly.

A standalone experiment on this PC tested this. Generated GLSL compiled with glslang, and one
`vkCreateComputePipelines` per module, no pipeline cache, idle priority. The sources are in
[rt-experiments/](rt-experiments/) and the method is in Appendix A.

| Shape | Blocks | Values | SPIR-V words | Pipeline creation | VGPRs | Scratch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Structured, short live ranges (L) | 128 | 1,024 | 84,831 | **0.48 s** | 23 | 0 B |
| Same blocks in a dispatcher (E) | 128 | 1,024 | 87,687 | **> 20 s (stopped)** | - | - |
| Structured, all values live (S) | 128 | 512 | 52,644 | 4.9 s | 256 | 2,592 B |
| Dispatcher, per-lane pc (D) | 128 | 512 | 54,888 | 13.8 s | 256 | 2,720 B |
| Dispatcher, uniform pc (U) | 128 | 512 | 54,909 | 17.5 s | 256 | 3,312 B |
| S / D / U | 64 | 256 | ~27k | 0.53 / 1.10 / 1.40 s | 256 | ~0.7-0.9 KB |

Conclusions:

- **The dispatcher itself, not SPIR-V size, is the problem.** With identical code, the dispatcher form
  is at least 40x slower to compile. The loop-carried liveness it creates pushes LLPC into the
  spilling regime.
- **Size and pressure still matter:** doubling size and pressure raised creation time 9-13x in every
  shape. Large structured kernels should stay near the guest's own register pressure.
- **Making `pc` wave-uniform does not help** (U vs D). Divergence analysis is not the bottleneck, so
  `subgroupBroadcastFirst` on the dispatcher `pc` is not worth doing.

### What the recompiler should emit instead

1. **Structured control flow for these kernels (Stage 2).** Normalize multi-exit loops in
   `ShaderCFG.cpp` before `SplitSharedMergeBlocks`. For each natural loop with exits other than its
   merge, make every exit edge a direct edge to one new loop merge `M'`. A phi at `M'` records which
   exit was taken (`exit_id`). After `M'`, a chain of structured selections on `exit_id` runs the
   original exit tails, which then fall through to the original merge. When one block leaves the
   loop by both edges, route one edge through a synthetic block and give the selection a dedicated
   unreachable merge. Process loops innermost-first, so multi-level breaks become one exit per
   level. The routing helpers already exist (`AppendGotoSelectBlock`, `AppendGotoSetBlock`,
   `ShaderCFG.cpp:1643-1668`). Then delete the `true_in_body != false_in_body` shortcut for loops
   that were normalized. Gate the pass on `decoded.has_bvh` until the journal diff (Stage 2) shows
   which other shaders it changes; see the performance rules in section 6.
2. **One intersection routine per module (Stage 3).** Add one IR opcode, `BvhIntersectRay` (result
   `U32x4`; operands: T# as `U32x4`, node pointer as `U32x2`, extent as `U32`, origin, direction and
   inverse direction as `U32x3` each, and EXEC). It replaces #640's roughly 300 inlined IR ops per call site, which every
   pass (SSA, constant propagation, DCE, SRT planning) must process. The SPIR-V backend defines one
   function, `kyty_bvh_intersect`, per module (in the style of `DefineGetBdaPointer`,
   `spirvEmitterMemory.cpp:932`) and calls it from each site. LLPC may inline the calls anyway;
   measure it. Even inlined, IR passes and SPIR-V size stay small, and the call site holds only a few
   live values.
3. **Cheap node fetch (Stage 3).** Today every flat dword load goes through `LoadBda`
   (`spirvEmitterMemory.cpp:210`). That means an EXEC test, a `get_bda_pointer` call (a page-table
   load and a presence branch), and an unaligned-dword path. A 64-byte node costs 16 of those. The
   routine should translate the node address once. BDA pages are 16 KiB (`CACHING_PAGEBITS = 14`,
   `bufferCache.h:30`), so a second lookup is needed only when a 128-byte node starts in the last
   128 bytes of a page. Then load `uvec4`s through a `PhysicalStorageBuffer` pointer with
   `Aligned 16` (4 loads for box16 and triangles, 8 for box32).
4. **Branch on node type** instead of computing both the box and the triangle result and selecting,
   as #640 does. Use `NMin`/`NMax` and explicit NaN tests so the results match GPURT's
   `fast_intersect_bbox` NaN rules.
5. **Keep the dispatcher as a last resort, but make it cheaper (Stage 5, optional).** Spill only
   values that are live across the dispatcher's back edge (a liveness pass), and confine the
   dispatcher to the irreducible region instead of the whole function. This helps every other
   dispatcher-fallback shader too, but changes their SPIR-V, so gate and measure it as in section 6.

### Could RDNA4's hardware RT be used instead (`VK_KHR_ray_query`)?

**Verified:** the driver exposes `VK_KHR_acceleration_structure`, `VK_KHR_ray_query`,
`VK_KHR_ray_tracing_pipeline`, `VK_KHR_ray_tracing_position_fetch`,
`VK_KHR_pipeline_executable_properties` and `VK_AMD_shader_info` (vulkaninfo, September 25).

Vulkan has no per-node intersection operation, and RDNA4's own node format (RTIP 3.1 quantized
BVH8) is private to the driver. A hardware path therefore needs **both**:

- **An acceleration-structure mirror of the guest BVH.** Walk the guest pool on the GPU, write each
  triangle node's vertices into a vertex buffer, and build one BLAS per guest BLAS root. Build a TLAS
  from guest instance nodes (once their layout is known). Keep a side table from
  (instance, geometry, primitive) to the guest triangle-node pointer. Rebuild or refit when the guest
  writes the pool; the buffer cache already tracks GPU-dirty ranges. Static BLASes build once.
  Dynamic ones (characters, physics objects) need a rebuild or refit every frame. For about 1-3M
  triangles, estimate 1-6 ms per full rebuild on this GPU, with refits cheaper (inferred; measure).
- **High-level emulation of the whole guest traversal loop.** Recognize, in each kernel, the loop that
  calls the BVH instruction, with its stack push/pop, instance transform branch, closest-hit update
  and any-hit tests. Replace it with `rayQueryInitializeEXT` / `rayQueryProceedEXT` and write the
  guest's hit registers in the guest's format (t, barycentrics, the guest node pointer from the side
  table). Any any-hit logic, such as alpha tests, must run in the proceed loop.

It is plausible (hardware traversal is roughly an order of magnitude faster than software), but it is
the largest and riskiest option:

- Hit results differ at edges (watertightness, box growth, tie order).
- Guest traversal variants multiply the pattern-matching work.
- Nothing works until the instance format and every kernel's loop are understood.

**Recommendation:** Stage 6, only if Stage 4 measurements show that software RT costs more than about
5 ms of GPU time per frame after Stage 5.

## 4. Question 3: where #558's 3 fps goes

**Reported data (theantipopau, #558 head `f9c1e3f`, RX 9070 XT):** 60 fps for about 100 s during boot,
then about 3 fps that stays even on the 2D logo. No thread above about 4% CPU. The GPU thread waits in
`MasterSemaphore::Wait` <- `CommandScheduler::Wait` <- `GuestGpu::ProcessCommands`. The 3D engine
averages about 50%. There were no long compile stalls.

**Budget at 3 fps (inferred):** a frame is about 330 ms, with about 165 ms of GPU work and about 165 ms of
GPU idle. The non-RT frame on the fork needs about 15 ms of GPU time (40% of a 37 ms frame), so about
150 ms per frame is RT-kernel execution, and the rest is CPU/GPU serialization.

GPU side (code reading, #558):

| Cause | Where | Fixable? |
| --- | --- | --- |
| Box nodes decoded with the #558 layout; if the data is RTIP 1.1, bounds and child pointers are garbage, so traversal wanders | `Translator::IMAGE_BVH_INTERSECT_RAY`, `BvhDecompressBound` | yes (Stage 1 + 3) |
| Triangle nodes always miss: no tmax shrink, no first-hit termination. Every ray visits every box it overlaps, the worst case | same, `is_box` select | yes |
| 16 dword `LoadAddressU32`s per node through the BDA path (in the fork's emitter, each one is an EXEC branch, a page-table load and a presence branch; #558's emitter may group them by four) | `RayTracing.cpp` node fetch | yes (Stage 3 item 3) |
| 270k-310k-word kernels; traversal inlined at every call site; probably 256 VGPRs plus scratch, with instruction-cache misses | #558 description | partly (Stage 3, 5) |
| Invalid SPIR-V (unmerged selections) accepted by LLPC; the generated code is unspecified | main's structurizer | yes (Stage 2) |

CPU/synchronization side (#558-specific unless noted):

| Cause | Where | In the fork? |
| --- | --- | --- |
| Full drain before every indirect dispatch, indirect draw and CondExec (`SyncGpuCleanBacking`) | `96b457e4` in `pm4Handlers.cpp`, `graphicsRun.cpp` | **no:** the fork runs indirect dispatches from the GPU buffer (`RenderExecutor::DispatchIndirect`, `renderCompute.cpp:404`) |
| Occlusion queries: `DrainOcclusionBeforeReuse` waits for the previous tick in every `BeginNext`, plus `BufferFlushAndWait` for wait-predication | `7b71a647` | no (the fork keeps synthetic visibility) |
| Guest-thread read faults serviced through `ProcessCommands` as full drains | shared code | **yes:** the fork's largest remaining drain (about 9.5 ms/frame on the overworld, `performance-amd.md`) |

With RT kernels in the queue, each full drain waits for them, so CPU and GPU time add up instead of
overlapping.

**Is shader traversal inherently too slow? Probably not (inferred; Stage 4 measures it).**

- Assume about 400-600 lane operations per visited node, covering the intersection and the recompiled
  guest loop body, and 50% SIMD efficiency. The RX 9070 XT (64 CUs, about 2.9 GHz, 64 lanes/clock/CU
  without dual issue) then visits about 10-15 billion nodes per second.
- Digital Foundry's review mentions no RT reflections or shadows, and the game holds 60 fps at
  1440p-2160p on PS5, so its RT is probably GI-scale. At GI scale, 0.1-1M rays per frame x 30-60 nodes per ray is 3-60M
  node visits, or **0.3-6 ms** of GPU time.
- Memory traffic is 64-128 B per node. Upper tree levels stay in the 64 MB Infinity Cache.
- PS5 hardware peaks at about 320 billion box tests per second (4 per clock per CU x 36 CUs x 2.23 GHz),
  but real RT is latency-bound far below that. Expect software on this GPU to be a few times slower
  than the PS5 on the RT portion, not 100 times.

If Stage 4 measures more than about 8 ms per frame for correct traversal, the remaining options are
Stage 5 work and then Stage 6.

## 5. Question 4: which effects use ray tracing

Known (verified from `C:\Games\_Patches\PPSA21564.json`):

- **Mod 1, "Select the existing non-tiled deferred-lighting renderer":** at eboot `0x73b3bcf`, a
  `test r14b,r14b; je` becomes an unconditional `jmp`, which always takes the non-tiled path. The
  tiled deferred-lighting path therefore includes RT work. Candidates are ray-traced light
  visibility, AO or reflections inside tile lighting (inferred).
- **Mod 2, "Disable GI probes and lighting shaders":** at `0x70cd9c3`, `movzx eax, byte [rdx+0x7a0]`
  becomes `mov eax, 0`. This copies zero into a settings field (`[rdi+0x5e4]`) instead of the
  configured flag. GI probe updates are therefore at least one RT consumer. The flag's neighbours
  (`[rdx+0x7a0]`, `[rdi+0x5e4]`) may hold more feature flags. They can be inspected statically in
  the decrypted eboot with a disassembler, without running the game.

The PRs name eight kernels containing `0xE6` (#558) and four seen by the title screen (#640). Three
hashes are public: `503d7f066a496c3c`, `1f16e50eea0c89e3`, `ff8ee744ffa4dcdc`.

**Enabling one effect at a time:** yes, at two levels.

1. **Game level:** the two mods are independent, which gives four configurations:
   - both on (today)
   - mod 2 off: GI probes on, non-tiled renderer
   - mod 1 off: tiled renderer, probes off
   - both off: everything
2. **Emulator level (Stage 0):** `--rt-kernels <hash,...|all>` emulates only the listed BVH kernels
   and keeps skipping the others. Skipping part of a multi-pass effect can leave stale or garbage
   inputs, so prefer switching effects at the game level, and use the allow list for bring-up.

**How the census works (Stage 1 session):** run each configuration with `--ray-tracing capture` and
`--rt-stats`. For each kernel, record dispatches per frame (direct or indirect) and what it writes:
storage-image formats and sizes, and buffer ranges, from the captured specialization. Classify with
simple rules:

- a small 2D-array or octahedral-tiled atlas means probe GI
- a screen- or tile-sized R8/R16 target means visibility or AO
- screen-sized RGB means reflection or lighting

Screenshots of each configuration with the kernels skipped, compared with PS5 footage, show what is
missing.

## 6. Performance rules for all stages

The user's priority is that nothing makes the default (patched, RT-skipped) path slower.

1. **Default `--ray-tracing skip`:** detection stays at decode (`ShaderDecoder.cpp:415`), and the skip
   entry stays in `ProgramCache::Get` (`pipelineCache.cpp:289`, `:357`). A skipped dispatch costs the
   same map lookup as today. Keep the recompiler free of `Config`; pass the mode in `CompileOptions`
   (`ShaderRecompiler.h`) as PR #493 recommended.
2. **Gate the recompiler changes** (loop-exit normalization, SrtWalker loop-carried reads,
   dynamic-buffer lowering) on `decoded.has_bvh` until a **journal diff** shows zero changed SPIR-V
   across the user's own shader corpus (`C:\Games\runs\fork\_PipelineCache\PPSA21564.shaders`,
   1.5 MB). Widen a gate only with a measured A/B.
3. **Stats sites** (`--rt-stats`) cost one relaxed load when off, following the `DrainStats` pattern
   (`drainStats.h`).
4. **Caches:** the driver cache and journal are keyed on a SHA-256 of recompiler and pipeline sources
   (`performance-amd.md`). Any recompiler commit invalidates both once, so the first launch after
   updating recompiles and hitches. This is not a steady-state cost. Do not merge RT recompiler
   commits into the play build while another session is running warm-cache A/B measurements.
5. **Re-measure the patched configuration** (same save, route and warm cache, `--drain-stats`) before
   and after each stage lands. The expected result is no change.

## 7. Staged plan

Sizes are rough (production C++ + tests). **[machine]** marks steps that need the user's PC for the
game or significant GPU/CPU time. Everything else is code reading, CPU-only unit tests, or offline
analysis.

### Stage 0: plumbing and counters (1-2 days, about 300-450 lines)

- `src/common/emulatorConfig.{h,cpp}`, `src/main.cpp`, launcher: `--ray-tracing <skip|capture|emulate>`
  (default skip), `--rt-kernels <hash,...|all>`, `--rt-stats <seconds>`. Start from
  `.tools/kytyps5/notes/raytracing/raytracing-setting-plumbing.patch` (Skip/Emulate) and add Capture.
- `CompileOptions::bvh_mode`; `TranslateProgram` (`ShaderRecompiler.cpp:529`) skips only in skip
  mode or for kernels not on the allow list.
- `rtStats.{h,cpp}` next to `drainStats`: per shader hash, count skipped and executed dispatches
  (direct/indirect) and pipeline-creation milliseconds. In emulate mode, add GPU time from timestamp
  queries around BVH dispatches only (`RenderExecutor::DispatchDirect`/`DispatchIndirect`,
  `renderCompute.cpp:198`, `:404`).
- Tests: the existing `TestRayTracingDispatchDetection` must pass unchanged (skip mode). Add: capture
  and emulate modes do not skip at `TranslateProgram` (until Stage 3, emulate uses a stub that
  returns all ones, as in #493), and the allow list parses hashes.

### Stage 1: capture and offline tools (3-5 days, about 900-1,300 lines)

- Decode `0xE6`/`0xE7` in `frontend/decode/ImageOps.cpp`: address counts 11/12 (8/9 with A16),
  R128/DIM/DMASK checks, NSA. Port from #558 `4ccbe842` or #640 (keep authorship where code is
  taken).
- Capture as described in section 2 (`rtCapture.{h,cpp}`), plus T# plan sources.
- `ShaderPrecompile::ReadRecordsUnchecked(path)` (a tool-only reader that ignores the ABI key) and a
  **journal diff tool**. For every record: `TranslateProgram` + `CompileProgram` with the recorded
  specialization. Print hash, stage, CFG mode and reason, SPIR-V words, `OpVariable`/`OpPhi`
  counts, and the SPIR-V SHA-256. Compare two builds' outputs. Put it in
  `tests/ShaderPrecompileRecordTests.cpp` as a `--journal-report <file>` mode. It is CPU-only and
  takes seconds.
- `bvh_inspect` analyzer (section 2), as `shader_recompiler_compute_tests --bvh-inspect <dir>`,
  CPU-only.
- **[machine] Session A (about 30-45 min of the user's time, after one build):**
  1. Patched and warm, with `--rt-stats`: confirm zero BVH dispatches and unchanged fps.
  2. Capture with mod 2 off: title screen, then the overworld for about 2 min.
  3. Capture with mod 1 off.
  4. Capture with both off.
- **Offline results:** node-format verdict, instance layout, kernel census per switch, guest code for
  every BVH kernel, and each kernel's CFG outcome on the current structurizer.

### Stage 2: structured control flow for the RT kernels (1-2 weeks, about 350-600 lines plus about 500 lines of tests)

- Loop-exit normalization in `ShaderCFG.cpp` (section 3, item 1), gated.
- Tests in `tests/shaderCfgTests.cpp`: a loop with one exit tail that rejoins, two tails, a tail
  that ends the program, an inner loop that breaks straight out of the outer loop, and one block
  leaving by both edges. Each must structurize without the dispatcher and pass `spirv-val`
  (`CheckSpirvBinaryValidates`). Also check that shapes which already structurize produce identical
  graphs, whether or not the pass is enabled.
- Offline: all captured RT kernels must structurize (from the journal report). Record SPIR-V words
  and variable/phi counts.
- **[machine, about 1-5 min of CPU, ask first]:** time `vkCreateComputePipelines` for each captured
  kernel with the `rt-experiments` harness, loading SPIR-V produced by the journal tool. Record time
  and `VK_KHR_pipeline_executable_properties` statistics (VGPRs, SGPRs, scratch). **Target:** under
  about 5 s per kernel cold. Journal replay and the driver cache hide it on later runs.

### Stage 3: correct instruction emulation (1.5-2.5 weeks, about 1,000-1,400 lines plus about 1,000 lines of tests)

- Frontend: `frontend/translate/RayTracing.cpp` emits `BvhIntersectRay`. Read the T# as raw SGPRs
  (never as a buffer resource; #555's crash). Unpack A16 operands in the frontend.
- IR plumbing: `ValueOpcodes.inc`, DCE and constant propagation treat it as a pure memory read.
  `ResourceTracking` sets `uses_dma` only. Keep the T# dwords runtime values (flattened SRT slot or
  dynamic read), never specialization constants, or every pool reallocation would compile a new
  permutation.
- Backend: new `spirvEmitterRayTracing.cpp` with `kyty_bvh_intersect` (section 3, items 2-4):
  - box16, box32 and triangle types 0/1, plus types 2/3 once Stage 1 shows their vertex selection
  - triangle_return_mode 0 and 1, box_sort_en, box_grow_value, BVH64 addressing
  - types 6/7 and out-of-range pointers return all ones
- Port the prerequisites, gated on `has_bvh`:
  - #558 `fb3b2795` (SrtWalker: no flat slot for a raw read whose base is loop-carried). Refine it to
    phis that `ResolveInvariantPhi` cannot resolve, so shaders that work today are unchanged
    (`SrtWalker.cpp:372`, `Collect`). Without it, `MaterializeResources` fails and
    `pipelineCache.cpp:298`/`:365` `EXIT_IF`s.
  - #558 `21ca9881` / #640 `DynamicBuffer.cpp` (loop-carried V# -> raw address loads).
- A C++ golden model in the tests: a line-by-line port of GPURT `image_bvh64_intersect_ray_base`,
  `IntersectNodeBvh4`, `fast_intersect_bbox`, `fast_intersect_triangle` and `SwizzleBarycentrics`.
- Tests (section 8). **[machine, seconds of GPU, ask first]:**
  `shader_recompiler_compute_tests --bvh-only`.
- Measurements, offline: SPIR-V words per call site (target under 300 plus one shared function) and
  per kernel. **[machine]:** GPU microbenchmark of 1M rays against a synthetic 100k-triangle BVH
  (timestamp queries), giving node visits per second on this GPU. That checks the section 4 estimate
  before the game is involved.

### Stage 4: one effect at a time in the game (3-5 days plus sessions)

- Emulate mode with `--rt-kernels` for the probe-GI kernels first (mod 2 off, mod 1 on). This is
  likely the smallest and most error-tolerant effect (low-frequency lighting).
- **[machine] Session B (about 30-60 min):** for each configuration, with emulation and with skipping
  (A/B):
  - fps
  - `--drain-stats` (drains per frame, ms)
  - `--rt-stats` (GPU ms per RT kernel, dispatches per frame, compile ms on the cold run)
  - the GPU-side trace for about 1 s, checked offline against the golden model
  - screenshots of the same spots
  - optionally a Radeon GPU Profiler capture of a few frames
- Then the tiled-lighting kernels (mod 1 off), then everything.
- **Exit criteria:** correct-looking lighting, and no dispatcher fallback in any RT kernel. Also a
  per-frame RT GPU-time figure, which decides whether Stage 5, Stage 6 or neither is needed.

### Stage 5: make it fast (1-3 weeks, open-ended; driven by Stage 4 numbers)

- **Drains:** RT kernels make every full drain costlier. Asynchronous guest-fault readback
  (`roadmap.md`, medium item 1) becomes more valuable, as does a pending-download tick instead of
  `Wait(CurrentTick())`.
- **Node fetch without the page table:** at dispatch, evaluate the T# (base, size) and bind the pool
  as one buffer-cache buffer whose device address goes in push data, with a fallback to BDA when the
  T# cannot be evaluated. This also pre-registers the pool, so the first frame after a rebuild does
  not read zeros through `RecordBdaFault`.
- **Register pressure and occupancy:** use `VK_AMD_shader_info` or pipeline-executable statistics
  per RT kernel. Shrink the code the recompiler emits for the hot guest loop body; the
  per-instruction bloat is shared with all shaders.
- **A cheaper dispatcher**, for any remaining shaders (section 3, item 5).
- **Hitch control:** make sure journal replay precompiles RT kernels at startup (it does for recorded
  permutations), so a cold RT kernel appears only on the first run.

### Stage 6 (optional research): hardware ray queries (4-8+ weeks)

As in section 3. Do this only if software RT still costs more than about 5 ms/frame after Stage 5.
Prototype on one kernel with static geometry first.

### Machine-time schedule

| When | Needs | Length | Blocks |
| --- | --- | --- | --- |
| After Stage 0+1 code | one full build (4-8 min, ask) + Session A | about 45 min | Stage 1 verdict |
| During Stage 2 | journal diff (CPU seconds); pipeline timing of captured kernels (about 1-5 min CPU, ask) | minutes | Stage 2 sign-off |
| During Stage 3 | test build (about 80 s, ask) + `--bvh-only` + microbenchmark (seconds of GPU) | about 5 min | Stage 3 sign-off |
| Stage 4 | build + Session B | about 1 h | go/no-go on performance |
| Each merge into the play build | patched-configuration A/B (section 6.5) | about 15 min | none |

## 8. Tests to add

In `tests/ShaderRayTracingTests.inc` (included from `shaderCfgTests.cpp`, CPU-only) unless noted:

1. **Decode:** every captured encoding, including #640's `f1989f07 00040505 ...`, decodes with the
   right address count (11/12/8/9), NSA register order, T# SGPR base and VDATA. Reject R128=0,
   DMASK != 0xF and DIM != 1D.
2. **Modes:**
   - skip: today's behavior; the existing detection tests stay unchanged
   - capture: translates, but `ProgramCache` still skips
   - emulate + allow list: only listed hashes translate
3. **IR:** exactly one `BvhIntersectRay` per instruction; the T# never becomes a buffer resource or a
   specialization constant; `uses_dma` is set; DCE keeps the op when its result is used.
4. **Golden model vs emulation** (CPU IR evaluation, as #640's `TestBvhIntersections` does): box16
   and box32 hits and misses, ray origin inside a box, zero and negative direction components with
   infinite inverse direction, NaN handling, box_grow, box sorting on and off, extent clipping, child
   `0xFFFFFFFF`, triangle hit and miss, degenerate triangles, both return modes, barycentric swizzle
   from triangle_id, types 6/7, BVH64 carry across 32 bits, and inactive lanes (no memory access).
5. **Structurizer:** the loop-exit cases from Stage 2, each passing `spirv-val`. A regression check
   that previously structured graphs are unchanged.
6. **SrtWalker:** a raw read with a loop-carried non-invariant base goes to `dynamic_reads`; one with
   an invariant phi keeps its flat slot (`ResourceMaterializationTests.cpp`).
7. **GPU** (`tests/ShaderRecompilerComputeTests.cpp`, `--bvh-only`): random rays against synthetic
   box16/box32/triangle nodes, compared bit-exactly for pointers and within 1 ULP for floats. Plus a
   full traversal: a small guest-style traversal loop encoded in RDNA2 instructions (stack in LDS)
   over a synthetic 1,000-triangle BVH, compared with a CPU brute-force closest hit.
8. **Journal tool:** round-trips a record; the report is stable across two runs.

Synthetic BVHs come from a tiny in-test builder that writes the confirmed node formats. Game data is
never used in the repository.

## 9. Risks and open questions

- **Format not RTIP 1.1 after all:** Stage 1 would show it immediately. The emulator design does not
  change, only the decode in `kyty_bvh_intersect`.
- **Triangle types 2/3 and the instance layout** are unknown until capture.
- **Structurizer complexity:** loop-exit normalization is subtle. Keep it gated. If a kernel still
  cannot be structured, a region-local dispatcher (Stage 5) limits the damage.
- **Result differences:** hardware watertightness and rounding cannot be matched bit for bit without
  PS5 hardware. GPURT's software reference is the best available oracle. Probe GI tolerates small
  differences.
- **RT scale unknown:** if Astro Bot traces per-pixel rays in the tiled path, the section 4 estimate
  may be too low. Stage 1's census gives dispatch sizes before any emulation work is done.
- **Overlap with other work:** `pipelineCache.cpp`, `SrtWalker.cpp` and `ResourceMaterialization.cpp`
  are also being changed by the per-draw materialization session. Land RT recompiler changes after
  it, and rebase.
- **Upstream:** #640 and #558 are active. The findings here (GFX1013 is `RT_1_1`; the dispatcher
  compile-time experiment; loop-exit normalization) could help both. Posting them is the user's
  decision.

## References

Local reference copies are in `C:\Users\Bryan\Documents\Chat\.tools\kytyps5\notes\raytracing\ref`:

- AMD GPURT: `IntersectCommon.hlsl` (`image_bvh64_intersect_ray_base`, `IntersectNodeBvh4`,
  `fast_intersect_bbox`, `fast_intersect_triangle`, `SwizzleBarycentrics`), `TraceRay1_1.hlsl`, and
  `gpurt/src_shadersClean_common_gfx10_*` (`BoxNode1_0`, `TriangleNode1_0`, `InstanceNode1_0`),
  `..._ShaderDefs.hlsli` (node types), `..._NodePointers.hlsli`, `..._Common.hlsl`
  (`CalcTriangleVertexOffsets`), and `..._gfx12_QuantizedBVH8BoxNode.hlsli`.
  https://github.com/GPUOpen-Drivers/gpurt
- PAL: `gfx10_sq_ko_reg.h` (`sq_bvh_rsrc_t`) and `gfx9Device.cpp` (`Device::CreateBvhSrds`).
  https://github.com/GPUOpen-Drivers/pal
- Mesa: `radv_nir_rt_common.c` (ref copy). `src/amd/common/ac_gpu_info.c`, lines about 341-342 and
  929-930 as fetched on September 25, 2026:
  https://gitlab.freedesktop.org/mesa/mesa/-/blob/main/src/amd/common/ac_gpu_info.c
- Mesa MR 33116 (Cyan Skillfish / BC-250):
  https://gitlab.freedesktop.org/mesa/mesa/-/merge_requests/33116 and
  https://www.phoronix.com/news/AMD-RADV-PS5-BC-250,
  https://www.phoronix.com/news/Mesa-25.1-RADV-AMD-BC-250
- LLVM `SIISelLowering.cpp` (`amdgcn_image_bvh_intersect_ray` operand layout).
- AMD RDNA2 ISA (MIMG opcodes 230/231, Table 48):
  https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
- Upstream PRs (snapshots `pr*.json`, `c558.json`, `c640.json`, `rt558.cpp`, `rt640.cpp`; refs
  `review/pr-555`, `review/pr-558`, `review/pr-640`, `review/pr-747`):
  - https://github.com/KytyPS5/KytyPS5/pull/640: software RTIP 1.1, CFG change, AMD compile times
  - https://github.com/KytyPS5/KytyPS5/pull/558: Astro Bot branch, SrtWalker fix, 3 fps report
  - https://github.com/KytyPS5/KytyPS5/pull/555: T# classification crash
  - https://github.com/KytyPS5/KytyPS5/pull/493: decode + opt-in stub
  - https://github.com/KytyPS5/KytyPS5/pull/747: duplicate of #558
  - https://github.com/KytyPS5/KytyPS5/issues/281: titles blocked on `0xE6`/`0xE7`
- Digital Foundry, Astro Bot tech review: https://www.youtube.com/watch?v=1qhV6Tv1FkM (no RT effects
  mentioned in the forum summaries; the video itself was not checked; summary via https://www.resetera.com/threads/digital-foundry-astro-bot-on-ps5-virtually-flawless-digital-foundry-tech-review.972954/)
- Fork docs: [performance-amd.md](../performance-amd.md), [roadmap.md](roadmap.md),
  [raw-findings.md](raw-findings.md).

## Appendix A: compile-time experiment

Question: on this driver, does the dispatcher shape itself, rather than SPIR-V size, cause the
multi-minute compiles?

- **Generator:** [rt-experiments/dispatcher_shape_gen.cpp](rt-experiments/dispatcher_shape_gen.cpp).
  It writes GLSL compute shaders whose blocks each do six multiply/add or xor/shift statements on
  `uint` locals, with 1 KiB of loads at the start and a store of the combined result.
  - D, U and S keep all values live to the end.
  - L and E give each block 8 new values computed from the previous block's 8.
  - D and E use `while (pc != ~0u) switch (pc)` with per-block next-pc selection, as
    `EmitDispatcherFunction` does. In E, next-pc depends on a loaded value, so the compiler cannot
    fold the case order.
  - U adds `subgroupBroadcastFirst` on `pc`.
- **Harness:** [rt-experiments/vk_pipeline_time.cpp](rt-experiments/vk_pipeline_time.cpp). It loads
  `vulkan-1.dll`, picks the device whose name contains "9070", enables
  `VK_KHR_pipeline_executable_properties`, and times one `vkCreateComputePipelines` (no
  `VkPipelineCache`, `CAPTURE_STATISTICS`) at `IDLE_PRIORITY_CLASS`. It prints the driver's
  statistics.
- **Build and run:**
  - Build with clang-cl 23 from `.tools/kytyps5` (`kyty-env.ps1`) and the repository's
    `3rdparty/Vulkan-Headers`.
  - Compile the shaders with `glslang -V --target-env vulkan1.3`.
  - Each run used a new seed in the initial values, so the driver's own disk cache could not serve a
    result.
  - Each compile had a 20 s `timeout`.
  - Total CPU time was about 70 s on one core at idle priority.
- **Results:** the table in section 3. The 256-block / 1,024-value D/U/S modules were generated but not
  timed, to keep the run short.
- **Caveat:** these are synthetic shaders. The real kernels should be timed in Stage 2 with the same
  harness on SPIR-V produced from the Stage 1 capture.
