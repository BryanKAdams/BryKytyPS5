# AMD performance work

This branch starts from `83628a436e58a52814a5322dd5ba91d4d86479d2` and adapts the remaining
performance proposals to that revision's renderer. The development machine is a Windows PC with
a Ryzen 7 7800X3D and Radeon RX 9070 XT. Astro Bot gameplay and a 60 fps result have not been
validated by this patch set.

## Implemented changes

- **Predicate synchronization (PR #702):** bool predicates skip submission when their bytes have
  no GPU producer. Buffer producers download the required range; image producers and deferred
  image writebacks retain conservative GPU and priority-operation completion. Overlapping pending
  image writebacks are tracked independently until their backing-memory writes finish.
- **BDA discovery (PR #562):** CPU-dirty region hints narrow the search for cached buffers requiring
  uploads. The per-page dirty state remains authoritative. Region creation, CPU writes, mapping,
  and buffer registration publish hints; concurrent publications are preserved across consumption.
  Inconsistent ownership falls back to the legacy walk. A 64-word summary (one bit per hint word)
  lets a pass skip clean hint words, so an idle pass reads 64 words instead of 4,096.
- **Windows address waits (PR #618):** native `WaitOnAddress` parks stack-allocated waiter records.
  Sharded locks replace the shared allocating registry. Explicit wakes remain visible during signal
  callbacks, and counted wakes are bounded by registered waiters rather than the requested count.
- **Shader replay (PR #718):** a versioned, checksummed journal records shader code and compile
  metadata, without runtime pointers or buffer addresses. Warm startup replays the journal before
  the first shader lookup. Runtime resources are rematerialized from the current guest state.
- **Readback and pressure handling (PR #638):** downloads exceeding the 64 MiB staging ring use
  dedicated staging allocations retained through publication. Pressure collection retries image
  lookup after reclaiming eligible textures. Dirty victims remain protected until their writebacks
  complete. Overlap replacement also keeps the old image registered and write-protected until
  pending publication completes; the wait releases the texture lock and does not run unrelated
  deferred resource-destruction callbacks. Safely downloadable depth planes retain their eviction
  path when their stencil companions are known clean; dirty or unknown stencil ownership, metadata,
  and conflicting alias cases remain resident.
- **Mesh dispatch limits (PR #793):** oversized mesh draws are sliced to the host's per-axis and
  total limits. Primitive and instance offsets survive slicing. The normal draw path visits slices
  without allocating a vector.
- **Frame reporting:** the window reports fresh game frames separately from presentation rate,
  excludes reused frames from game FPS, and updates the title at most once per second.
- **AMD attachment feedback:** layout support is independent from dynamic feedback-state support.
  Drivers exposing the layout extension alone use static pipeline flags keyed by actual overlapping
  attachment reads/writes. Drivers supporting dynamic state continue to reuse one pipeline variant.
  The tested RX 9070 XT driver exposes the layout extension but not the dynamic-state extension.
- **Scalar mask preservation:** 64-bit SAVEEXEC operations retain raw EXEC words independently of
  host-active lanes, while preserving the saved per-invocation predicate for later restores. The
  complete Radeon shader suite exposed loss of inactive bits through the previous ballot path;
  existing expected register values remain unchanged. Scalar VCC/EXEC branches use the complete
  guest-wave zero flags, with the upper half included only in wave64; per-lane VALU masking is
  unchanged.
- **FP16 fractional boundary:** finite `V_FRACT_F16` results stay below one before destination
  modifiers are applied. The test's previous rounded-one expectations contradicted the DX
  fractional range referenced by AMD's RDNA2 ISA, section 12.8. Edge tests now require the largest
  half below one, while separate modifier checks allow multiplication and saturation to reach one.
  References: [AMD RDNA2 ISA](https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture),
  [Microsoft frac](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-frac).
- **FP16 inline constants:** use the fixed architectural half value for `1/(2*pi)` (`0x3118`,
  RDNA2 ISA section 6.2/Table 20). It must not depend on the host GPU's conversion of the distinct
  single-precision inline constant. Packed selectors and source modifiers retain their behavior.
- **Guest-wave buffer indexing:** ADD_TID offsets wrap at the guest wave width, including wave32
  shaders executed in a host subgroup of 64. Specialized and dynamically selected buffers share
  the corrected lane calculation.
- **Windows regression repairs:** the filesystem test has a valid SDL entry-point signature and
  now exercises Winsock receive-flag translation. Blocking stream `PEEK | WAITALL` is emulated
  without passing that invalid combination to Winsock; partial arrivals, deadlines, EOF and
  cancellation retain the bytes already received. Socket mode and timeout state are shared by
  transport. Per-call `DONTWAIT` on an otherwise blocking Windows transport remains unsupported
  and returns `EOPNOTSUPP`; an explicitly nonblocking transport supports it.

- **Per-draw shader resource refresh:** `ProgramCache::Get` refreshes every stage's descriptors
  on every draw and dispatch, including cache hits. Each `ResourcePlan` is now compiled once into
  index-based nodes, with identities, invariant phis, extract sources and immediates resolved.
  `SrtEvaluator` evaluates those nodes with the same contexts, clean predicates, EXEC-mask selects
  and failure cases as `SrtWalker`, but without `Value::Resolve` chains or per-use type dispatch.
  A per-entry memo skips descriptor evaluation and specialization when a plan's descriptors
  depend only on user data, the shader base and flat SRT slots, and those inputs and the active
  sources are unchanged. The flat SRT buffer, user data and uniform fill are still refreshed
  every time. Plans with dynamic reads, clean-only dependencies, `ReadFirstLane` or indirect
  images always take the full path; 276 of the 279 distinct Astro Bot plans qualify. Strict
  (GPU-clean) reads within one refresh are served from whole 64-byte blocks, which cuts
  `TryReadGpuCleanBacking` calls by about three times; a block that cannot be read whole falls
  back to exact reads. The permutation search checks the last hit first and is skipped when the
  memo kept the specialization and the push-data cursor is unchanged. The original walker is
  kept as `MaterializeResourcesReference`. Setting `KYTY_VERIFY_SRT=1` compares every refresh
  with it and aborts on a difference, and `KYTY_SRT_STATS=1` logs how often refreshes reuse
  descriptors.
- **Refresh reads (branch `perf/draw-cpu`):** at 60 fps `Thread_Gpu` was about 4% idle, with
  materialization at 29.6% of its samples. Direct SRT reads now come from 64-byte blocks
  (`TryReadGuestBlockOnGpuThread`), which copies a block only when no page it touches has a
  GPU-dirty hint. Otherwise the refresh reads that block's dwords with `ReadGuestOnGpuThread`,
  which keeps the exact-byte clean check. Branch conditions that read only flat SRT slots use
  those direct values, because every refresh reads the slots anyway. A condition is skipped
  when no block reachable from it guards a source. Descriptor dwords that are exactly flat
  slots come from the refreshed flat buffer. The strict reader no longer wraps every direct
  read in a forwarding call. Most Astro Bot conditions test constant-buffer words, so they
  keep their strict reads.

The dense evaluator, retained resource snapshots, and replacement of the old SRT readability path
were already present at the base revision. Their historical PR improvements are not additional gains
from this branch.

## Ported upstream pull requests

These open upstream PRs were brought in as separate commits that keep their original authors.

- **#483, skip the GPU drain on non-GPU unmaps:** `RenderContext::UnmapMemory` returns early
  when a range never intersected the GPU-mapped set. Unmapping 64 KiB of CPU-only memory from a
  guest thread took 79 us with an idle GPU and 130 us with 64 MiB of fill work queued; it now takes
  0.10 us.
- **#749, bounded event waits:** the main loop wakes at least every half vblank, so queued
  main-thread callbacks such as the title update cannot stall presentation. Ported to SDL3. In a
  1,200-update SDL3 harness, the unbounded wait's worst case reached 1.53 s (1.96 s in an earlier
  run); the bounded wait's worst case was 8.0 ms.
- **#820, zero reused direct memory:** new allocations and pool expansions no longer expose a
  previous owner's bytes. A follow-up commit clears only ranges that were returned to the free list
  before, so fresh backing is not touched (and committed) at allocation.
- **#761, fragment helpers in ballots:** helper invocations no longer count in guest ballots or
  in the lane-activity ballot for swizzle/bpermute. The PR's MoltenVK test-harness modes were not
  carried over; its tests run on the full harness.
- **#795, GPU tiler constants:** the tiler shaders avoid `uvec4` specialization-constant selects
  that RADV miscompiled on the RX 9070 XT. The Windows driver already passed the tiler tests.

## GPU drains in Astro Bot

`--drain-stats <seconds>` reports every host wait on the GPU by cause: full drains, waits on
submitted ticks, readback publication waits, blocked-queue polls, readback counts and DCC
metadata activity. Each row names the reason (for example a guest-thread read fault, a DCC
clear check, or a ring wrap) and the PM4 packet being executed. With the flag off, a wait site
costs one relaxed load.

The first Astro Bot captures (US 01.018, non-RT patch, RX 9070 XT) showed 12-18 ms of full
drains in each 33-40 ms frame. The largest source was the DCC fast-clear check. The game clears
its render targets by writing DCC keys from a compute shader every frame, and binding such a
target read the keys back on the CPU: about 5 full drains, 5-7 ms per frame. Every check found
a real clear.

**GPU DCC clears:** with `VK_EXT_conditional_rendering`, a small compute pass now checks the
GPU-written metadata slices, writes one predicate per clear code, and consumes cleared slices.
The candidate clears run under conditional rendering, so the CPU never waits for the keys. A
slice is checked again only after a new GPU write to it, and any later lookup of a checked
slice (for example sampling the target) skips the readback. Textures, video-out surfaces,
volumes and drivers without conditional rendering keep the CPU path, and `--dcc-gpu-clear false`
forces it. On the overworld (ship save, same spot, same build and warm caches), the new path ran
at about 24.5 fps with 13 ms of drains per frame (3 drains), against about 21 fps and 19 ms
(9 drains) on the CPU path. The fixed-clear GPU test runs Astro Bot's own key-fill shader
through both paths and requires identical results.

The planet in the overworld shows blotchy dark patches and the background a fine dot pattern.
Upstream `main` (5a705dd) renders the same artifacts, so they predate this branch.

**Asynchronous readback:** a guest thread that faults on GPU-written buffer memory no longer
drains the GPU thread. Thread_Gpu records the download and submits it. The faulting thread then
waits on that download's own timeline tick and publication, while Thread_Gpu keeps processing
PM4. The memory tracker arms the downloaded pages with a token and a tick, and publication
clears only pages still armed with its token. A GPU write recorded in between disarms a page, so
it stays protected until a newer download publishes it.

**Label submits:** a RELEASE_MEM that writes a plain label (no interrupt) submitted the command
buffer every time, hundreds of times a frame in Astro Bot. The label value is written to guest
memory when the packet is parsed, so the submit only kept the GPU fed. It now submits only when
the GPU has retired all earlier work. Labels with an interrupt still submit right away.

**Eager readback:** Astro Bot's "Draw Shadow" thread reads 8 bytes, 3-4 times a frame, that a
compute dispatch wrote about 0.75 frames earlier. Each read faulted, and its download queued
behind all the GPU work submitted since: about 6 ms per read, 21 ms per frame. Pages that
guest-thread reads fault on are now "hot" (at most 64). A recorded write to a hot page is
downloaded at the next command-processor flush point, and the next read usually finds it
already published.

**Queue thread:** `vkQueueSubmit` took about 13 ms of each frame on Thread_Gpu, the saturated
thread: about 330 submits at 18 us each, plus waits for the queue lock held by present. The
render scheduler now allocates the tick and hands the command buffer to a dedicated thread,
which submits in tick order. `--async-submit false` restores inline submission.

**Shader hash cache:** 305 of the 308 recorded Astro Bot shaders declare no hash, so
`GetShaderParams` hashed their whole code (about 4.4 KB) on every draw. The shader map entry now
caches the hash, and `AgcCreateShader` registering the address again drops it.

Overworld measurements use the ship save: title screen, cross, then cross on "SAVE DATA 1". The
ship hovers at its spawn above the first planet with no further input. The title screen renders
far faster and is not a valid benchmark scene. Each run is 60 one-second samples of the
window-title frame rate after a 20 s settle, with the scene checked at the start and end.

| Build | Overworld fps (mean / median) | GPU 3D busy |
|---|---|---|
| 84295af (GPU DCC clears) | 25.7 / 25.7 | 58% |
| + asynchronous readback | 30.6 / 30.5 | 59% |
| + idle-only label submits | 34.6 / 34.8 | 61% |
| + eager readback of hot pages | 34.4 / 34.4 | 61% |
| + queue thread for submits | 39.1 / 39.2 | 62% |
| + shader hash cached per registration | 40.4 / 40.3 | 62% |
| + clean reads on protected pages (A/B 39.8 -> 42.2) | 42.2 (mean of runs) | - |
| + GPU-built mesh indirect draws (same binary, flag off -> on) | 42.2 -> 51.4 | - |
| + batched RELEASE_MEM submits (A/B, interval 0 -> 2000 us) | 48.8 -> 57.9 | - |

Eager readback removed the 21 ms per frame of guest waits and cut readback traffic from about
23 MiB to under 1 MiB per 5 s. The frame rate did not move, because Thread_Gpu, not the guest
thread, bounded the frame.

**Clean reads on protected pages:** once a frame, Thread_Gpu stalled about 4 ms reading one of
Astro Bot's vertex-attribute tables, in `ShaderApplyAttribSemantics` and again in the SRT
walker. ThinLTO folding made the symbolizer blame `GetShaderParams`, and the disassembly showed
the real site. The table is not GPU-written. It shares a 4 KiB tracker page with GPU-written
data, so the whole page is protected. Game-thread write faults nearby download a 512 KiB window
that also arms the page. The buffer cache now tracks the exact bytes of downloads still in
flight. Thread_Gpu's own table reads peek at the page's GPU-dirty bit without a lock, and on a
dirty page they read clean bytes from the backing instead of faulting. In alternating A/B runs
(two 45 s rounds each), this took the overworld from 39.8 to 42.2 fps.

**GPU-built mesh indirect draws:** the last Thread_Gpu stall, about 4.7 ms per frame, was a
mesh-emulated `DRAW_INDEX_INDIRECT` reading arguments that a compute dispatch wrote earlier in
the frame. All of Astro Bot's indirect draws with GPU-written arguments are mesh-emulated, so a
plain `vkCmdDrawIndexedIndirect` path would not help. Downloading the arguments eagerly and
submitting right away only turned the drain into an equally long tick wait, because when
Thread_Gpu reaches the draw, the GPU has not yet run the dispatch.

Mesh shaders now load their seven draw dwords from a parameter record, through a device address
in push-constant dwords 0 and 1. Direct draws write one record per slice. For a GPU-args draw,
a one-thread compute pass (`mesh_indirect_args.comp`) writes the record and a
`VkDrawMeshTasksIndirectCommandEXT` from the arguments, and `vkCmdDrawMeshTasksIndirectEXT`
draws. It applies the CPU path's index-count clamp and the host workgroup limits, and the draw
registers the whole index buffer for the shader's index reads. With the same binary, the
overworld runs at 42.2 fps with `--gpu-mesh-indirect false` and 51.4 fps with it on (alternating
A/B, two 45 s rounds each: 41.3 / 51.2 and 43.1 / 51.6).
Full-resolution screenshots of both match, and drain stats show no full drains on Thread_Gpu.

**Batched RELEASE_MEM submits:** with Thread_Gpu no longer stalling, the overworld held at about
51 fps, and no thread was saturated. GPU timestamps (`--drain-stats` now reports GPU busy time and
gaps) showed 11.8 ms of GPU work and 8.4 ms of gaps in each 20 ms frame, spread over about 340
command buffers. Almost all of them came from RELEASE_MEM packets carrying an interrupt, which
submitted every time. The queue thread spent 16 ms a frame in `vkQueueSubmit`, and the GPU
waited between tiny buffers. The game loads a compute-written float every frame (it does
`vmovss xmm0, [r8]` and stores the value into its state; it is not a fence), so late GPU work
bounded the frame. RELEASE_MEM submits now wait at least `--label-flush-interval-us` (default
2000) since the last submit; the end of every submission slice still submits. Alternating A/B
runs gave 48.8 fps at 0 and 57.9 fps at 2000 us. Nearby values were 1500 us: 56.5 and
2500 us: 57.2, while 5000 us fell to 43.5 because the game waits on the delayed interrupts.
Submits drop to about 20 a frame, and GPU gaps to about 0.6 ms. Three other experiments did not
move the frame rate, which stayed near 51 fps:
- `--gpu-frames-ahead` (opt-in): the suspend point waits only for an earlier frame's work.
- A 100 Hz virtual vblank.
- Merging the SRT materialization work.

Reviewed but not ported: #506 (its texture-residency change would raise memory to the pressure
threshold, where eviction drains the GPU; read-only compute barriers almost never apply; block
descriptor reads are already in `main`), #767 (duplicates #702 and the dense memo), #628, #484,
#473 and #420 (superseded in `main`), #373 (covered by #638), and #637/#735 (MoltenVK-specific).

## Runtime comparison controls

```text
--async-submit true           Default: submit the GPU thread's work from a queue thread.
--async-submit false          Submit on the GPU thread (the previous behavior).
--gpu-mesh-indirect true      Default: build mesh-emulated GPU-args indirect draws on the GPU.
--gpu-mesh-indirect false     Read the arguments on the CPU (the previous behavior).
--label-flush-interval-us N   Default 2000: minimum microseconds between RELEASE_MEM submits (0 = previous).
--gpu-frames-ahead N          Default 0: frames the game may build ahead of Thread_Gpu at suspend points.
--bda-sync Selective          Default: use dirty-region discovery, with conservative fallback.
--bda-sync Legacy             Use the full mapped-buffer walk for controlled comparisons.
--bda-sync SelectiveChecked   Use selective discovery and check remaining dirty-page coverage.
--shader-precompile true      Default: record and replay compatible shader permutations.
--shader-precompile false     Disable journal recording/replay for comparison or recovery.
--dcc-gpu-clear true          Default: apply GPU-written DCC clears on the GPU.
--dcc-gpu-clear false         Read DCC keys back on the CPU (the previous behavior).
--drain-stats <seconds>       Report GPU waits by cause every N seconds.
```

The driver cache and shader journal require a Release build. They are keyed on a SHA-256 of the
recompiler and pipeline sources (`src/graphics/shader/**` and
`src/graphics/host_gpu/renderer/pipeline/**`, including uncommitted edits) and on the GPU/driver
identity; shader records also include the game version. Commits that do not touch those sources keep
a title's caches valid, so A/B builds can be measured warm; any change to them invalidates both files.
The journal detects checksum damage and interrupted writes, but is not an authenticated format for
importing arbitrary third-party shader records. Deleting a title's `.shaders` file or disabling replay
is the recovery path for a problematic record set.

The existing window dimensions do not establish the game's internal render resolution. The legacy
shader optimization selector has not been activated by this work. `--amd-cpu` remains an instruction
compatibility option rather than an X3D tuning switch.

## Validation commands

Validation on September 24, 2026: the combined Release/IPO `kyty_emulator` and `kyty_tests`
build succeeded, and all 49 registered CTest targets passed (51.03 seconds, serial execution).
GPU tests explicitly selected `AMD Radeon RX 9070 XT`, driver identifier `8389003`, including
the complete shader suite, static attachment feedback, pending-writeback races, shader replay
equivalence, and tiling/texture-cache tests. The focused wave-mask, FP16, and indirect-buffer
groups also passed during diagnosis. Logs are retained in `_Build/windows/ctest-final.log` and
`_Build/windows/Testing/Temporary/LastTest.log`.

This validation covers the changed mechanisms and the repository's regression coverage. It does
not establish an Astro Bot frame rate, full-game compatibility, or end-to-end warm startup for an
actual game's recorded shader set. The formatted-FP16 precision allowance is documented below.

Build with the repository's normal Windows prerequisites and Clang:

```powershell
cmake --build _Build/windows --target kyty_tests --parallel 8
$env:KYTY_TEST_GPU = 'AMD Radeon RX 9070 XT'
ctest --test-dir _Build/windows --output-on-failure
```

`KYTY_TEST_GPU` is an optional substring selector in the GPU test harness; it does not change the
emulator's `--gpu` selection. The harness prints the chosen adapter and mirrors the renderer's
optional attachment-feedback and provoking-vertex capabilities. Generic depth-backend tests choose
a supported four-byte depth/stencil backing rather than assuming D24S8 support. The tested Radeon
driver exposes D32S8 but does not expose D24S8.

Focused cases are `memory_tracker`, `sync_on_address`, `frame_statistics`, `mesh_dispatch`,
`shader_cfg`, `shader_precompile_record`, `performance_memory`, and `shader_precompile_gpu`.
The GPU tests exercise real buffer/image readbacks, pending-writeback ordering, dirty-region
equivalence, safe eviction, and live versus recorded shader execution. Windows fault tests issue
actual CPU stores while publication is held by a gate, covering ordinary collection, pressure
collection, overlap-style retirement, and live-image readbacks (including writes outside the image
on the same tracked page).
Depth-only eviction fixtures perform a real depth-only GPU clear; separately dirty stencil planes
must remain resident.

```powershell
.\_Build\windows\shader_recompiler_compute_tests.exe --bda-benchmark
```

This synthetic benchmark compares the original full mapped-buffer scan with selective discovery
over the same warmed, unchanged cached-buffer workload. The reference directly calls the original
range walk; it excludes hint maintenance added to the runtime `Legacy` comparison mode. There are
32 alternating batches of 64 passes per mode. Reported medians are medians of batch-average CPU
time per pass, not individual latency medians or game FPS. Do not run it while builds or other heavy
CPU tasks are active.

On September 24, 2026, this workload on the Ryzen 7 7800X3D / RX 9070 XT measured:

| Discovery path | Mean CPU microseconds/pass | Median batch-average microseconds/pass |
| --- | ---: | ---: |
| Original full scan | 16.443 | 16.315 |
| Selective hints with summary words | 0.110 | 0.111 |

There were 512 unchanged cached buffers and 2,048 passes per mode. Before the summary words, the
selective pass scanned all 4,096 hint words and measured 7.068 µs (median 6.986 µs) on the same
workload. That fixed cost made it slower than the full scan below roughly 250 cached buffers
(0.96 µs versus 6.8 µs at 32 buffers). With the summary words it is faster at every tested size
(32 to 8,192 buffers). The benchmark does not measure dirty uploads, shader cost, GPU execution,
frame time, or Astro Bot FPS. The GPU identity logged by the harness was `AMD Radeon RX 9070 XT`,
Vulkan driver value `8389003`.

### Shader resource refresh benchmark

```powershell
Copy-Item C:\Games\runs\drain\_PipelineCache\PPSA21564.shaders $env:TEMP\astro.shaders
.\_Build\windows\shader_recompiler_compute_tests.exe --srt-benchmark $env:TEMP\astro.shaders
```

The benchmark reads a copy of a shader precompile journal. Do not pass the game's live journal:
the emulator repairs or rewrites that file when it opens it. The benchmark translates every
distinct shader in the journal and extracts its resource plan. It then checks, on four
synthetic memory patterns and two user-data sets, that the compiled evaluator and the memo produce
exactly the reference walker's snapshot and specialization, including which plans fail. Finally
it times one refresh of every plan per pass: 48 rotating batches of 8 passes per variant, reported
as the median of batch averages. Memory comes from a fake reader (zeros or a hash of the address),
so these numbers exclude real guest-memory misses and the real cost of `TryReadGpuCleanBacking`;
the table below gives the read counts instead. A third argument (seconds) loops the compiled
path so that a sampling profiler can attach. Do not run it while builds, games or other heavy
CPU tasks are active.

On September 25, 2026, the 308-record Astro Bot journal (279 distinct plans; 276 memoizable)
measured on the Ryzen 7 7800X3D:

| Refresh path | Median µs per stage (4 patterns) |
| --- | ---: |
| Reference `SrtWalker` (before) | 4.04-4.21 |
| Compiled evaluator | 1.39-1.45 |
| Compiled + memo, unchanged inputs | 1.02-1.10 |
| Compiled + memo, inputs change every call | 1.43-1.51 |

Both paths made the same 9,972 direct reads per pass. Strict reads fell from 724-1,147
to 140-395 per pass with 64-byte blocks. The memo costs about 3% when every refresh misses, so it
pays off once more than about 12% of refreshes reuse descriptors; `KYTY_SRT_STATS=1` reports the
game's actual rate. The benchmark also reports that 305 of 308 records have no declared hash, so
`GetShaderParams` hashes their code (4.4 KB average, about 120-210 ns with XXH3) on every lookup.
It does not measure permutation search, key building, GPU execution, frame time, or Astro Bot FPS.

## Remaining rendering work

Formatted FP16 buffer stores still use `PackHalf2x16`, whose rounding mode is implementation-defined
in the [Vulkan SPIR-V environment](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html).
The component-conversion regression accepts only two adjacent finite encodings for its one
non-representable input; all other components remain exact. This verifies numeric conversion instead
of raw low-bit copying, not bit-exact PS5 formatted-store rounding. A native guest-format oracle is
still needed before changing this production rounding behavior.

Real occlusion queries are not implemented here. They require decoding the currently ignored
`DB_COUNT_CONTROL`, accumulating counters across rendering and submission boundaries, and publishing
coherent, asynchronously completed guest results. Replacing the synthetic visibility counters without
those semantics could incorrectly hide geometry.

Full BVH/ray-tracing emulation is also unchanged. The base revision skips compute dispatches containing
unsupported BVH intersections. A higher frame rate with missing effects is not evidence of accurate
Astro Bot rendering at 60 fps.

For a gameplay comparison, keep the game version, patch set, save, route, and rendering configuration
fixed. Warm each build's cache separately, compare fresh-frame times and GPU/CPU traces, and check
image correctness as well as loading and save behavior. The 60 fps target is 16.67 ms per fresh frame.
