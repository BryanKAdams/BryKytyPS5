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
  The replay translates and compiles the records on up to seven threads without the cache lock
  and adds them in order: a warm Astro Bot boot's first draw waited 17 ms for its 298 records
  instead of 1.35 s, and the longest boot frame fell from 2.2 s to 0.75 s.
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
  materialization at 29.6% of its samples. The strict reader has its own userdata
  (`SrtRuntime::specialization_userdata`). The 64-byte strict-read cache therefore no longer
  wraps every direct read in a forwarding call, which had a stack-cookie check. The profile had
  named that call `vector<pair>::_Emplace_reallocate`, because it had no public symbol. Branch
  conditions that read only flat SRT slots use those direct values, because every refresh reads
  the slots anyway. A condition is skipped when no block reachable from it guards a source.
  Descriptor dwords that are exactly flat slots come from the refreshed flat buffer, and
  single-dword `ReadGuestOnGpuThread` copies avoid a `memmove` call. Only 16 of the 1,125 Astro
  Bot conditions read nothing but flat slots, and 4 are skipped. The rest test constant-buffer
  words and keep their strict reads. Serving direct reads from 64-byte blocks was tried and
  removed: about one overworld run in three stayed at 51-54 fps instead of 58. A wider copy can
  fault on GPU-written bytes of a protected page that single dwords never touch.

- **Indexed VGPR reads (`V_MOVRELS_B32`/`V_MOVRELD_B32`):** each one compares M0 with every
  register index up to the shader's VGPR limit. Constant propagation now tracks the possible
  values of a U32 when they form a small set (small bit fields and masks, arithmetic on such sets,
  selects, phis). It folds `==`/`!=` against a constant outside that set, so unreachable
  chain links and their register operands disappear. Astro Bot's hottest pixel shader
  (`ef31694ed8d87754`) has 51 `V_MOVRELS` whose M0 is a 3-bit field times 5. Its SPIR-V went
  from 3,133 to 840 `OpSelect` and from 57,764 to 32,254 words. GPU zones in alternating runs
  measured it about 5-9% cheaper per run (0.97-1.01 ms to 0.88-0.95 ms), and
  `92b1436c8042e396` (52 `V_MOVRELS`) about 2% cheaper per pixel. Shaders without `V_MOVRELS`
  stayed within 1%. The AMD driver already folded most of these selects, so the saving is
  smaller than the SPIR-V suggests; the smaller modules also compile faster. The overworld fps is
  unchanged because dynamic resolution spends GPU headroom on render size.

- **EXEC-masked select chains:** a masked write becomes `select(exec, new, old)`, so a masked
  region that writes one register several times nests selects on the same predicate. Constant
  propagation now points a false-arm use on that predicate at the old value, and a lane-local use
  whose result only matters where the predicate holds at the new value; the intermediate selects
  and their bit casts die. Astro Bot's big shaders shrank 3-8% in SPIR-V words
  (`900aba8df9448d3d` 183,562 to 169,310; `01d6f21611218e74` 143,677 to 132,774). AMD driver
  compute pipeline creation fell from 1385 to 1268 ms and from 1006 to 852 ms. These are
  first-use costs (shader cache stutter), not per-frame ones.

- **Structurizer post-dominators:** the structurizer recomputes dominators and post-dominators
  after every merge split (27-62 times for a big shader). The post-dominator dataflow visited
  blocks in program order, the slow direction for a backward problem, and took 63-77% of a big
  shader's recompile. Visiting them in reverse reaches the same fixpoint in a few passes. Over all
  308 recorded Astro Bot permutations, the SPIR-V is byte-identical, `CFG::Structurize` fell from
  1318 to 249 ms, and a full recompile from 3471 to 2363 ms. The slowest shader
  (`01d6f21611218e74`) now recompiles in 141 ms instead of 386 ms. This is paid on every first
  encounter and again by the boot-time journal replay.

- **Uniform `V_READFIRSTLANE` / `V_READLANE`:** these were emitted as
  `OpGroupNonUniformShuffle` by `FindLSB(ballot(exec))` or by the SGPR lane. In pixel shaders,
  AMD compiles such a shuffle (and `OpGroupNonUniformBroadcast` with a non-constant lane) to
  `ds_bpermute` and treats the result as per-lane, even when the lane is in an SGPR. The
  waterfall key of Astro Bot's lighting loops then made everything downstream vector code:
  per-light `S_BUFFER_LOAD`s became vector buffer loads, the address math VALU, and the list walks
  EXEC-masked `v_cmpx` loops. Applying `OpGroupNonUniformBroadcastFirst` to the shuffle result is
  exact, because every lane already holds the same value, and it compiles to `v_readfirstlane`
  into an SGPR (7c38c759). AMD ISA, `--pipeline-compile-time` with `KYTY_PIPELINE_ISA`:

  | Shader | VGPRs | Vector buffer loads | `v_cndmask` |
  | --- | --- | --- | --- |
  | `5a10a907c8a64fc4` | 177 to 139 | 96 to 1 | 1772 to 1662 |
  | `969c5ed15d14c8fe` | 134 to 128 | 96 to 1 | 1481 to 1371 |
  | `9d7e6bca13da68ed` | 140 to 120 | 43 to 1 | 753 to 639 |
  | `6996d4e234bd5b8e` | 134 to 128 | 96 to 1 | 1444 to 1334 |

  The cost is about 150 more scalar instructions per shader, mostly `s_wait_kmcnt`. In warm GPU
  zones, `9d7e6bca` measured 17-20% cheaper per pixel and `5a10a907` 3.8% cheaper; the others
  stayed within the ±6% band that byte-identical shaders showed. A compute shader with the same
  pattern (`d97f248195e22298`) had identical ISA either way, so check the hot pixel shader itself.

- **Branch-free storage-buffer loads:** every buffer and constant-buffer dword load sat in a
  branch on its bounds check, itself inside the EXEC branch. The load is now issued at its own
  index and only its value is replaced (zero, or the format's out-of-bounds value), which is what
  the branch gave. The device always enables `robustBufferAccess`, so an out-of-bounds load is
  harmless. LDS, scratch, stores, atomics and buffer-device-address loads keep their branches.
  The AMD compiler's cost follows control flow far more than SPIR-V size: merging duplicate pure
  instructions cut 6% of the words but only 1-2% of `vkCreateComputePipelines` and was dropped,
  while removing the bounds branches cut `900aba8df9448d3d` from 1397 to about 1180 ms. A first
  version clamped the index instead (`load(in ? index : 0)`). That blocked constant-offset
  folding into `s_buffer_load`/`buffer_load`, and warm GPU zones measured `ef31694ed8d87754`
  17.8% slower per pixel. Keeping the address fixed it. The adopted form (b3ffaa5f), measured in
  cold new-game runs with matched pipelines: graphics pipeline creation 15.2 to 13.0 s (-14%),
  compute 5.87 to 5.40 s (-8%), worst frame 3.8 to 3.3 s. Warm, standing still at the Crash Site
  start: GPU busy unchanged (10.54 vs 10.55 ms), and every zoned shader was 3.8-16% cheaper per
  pixel (`5a10a907c8a64fc4` -9.5%, `969c5ed15d14c8fe` -10.1%). The remaining branches in compute
  and mesh shaders mostly guard stores: 181 storage stores in `01d6f21611218e74`, and 249-307
  LDS stores in the big mesh shaders.

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
slice (for example sampling the target) skips the readback. Video-out surfaces, volumes,
textures whose image cannot be a color attachment, and drivers without conditional rendering
keep the CPU path, and `--dcc-gpu-clear false` forces it. On the overworld (ship save, same
spot, same build and warm caches), the new path ran at about 24.5 fps with 13 ms of drains per
frame (3 drains), against about 21 fps and 19 ms (9 drains) on the CPU path. The fixed-clear
GPU test runs Astro Bot's own key-fill shader through both paths and requires identical
results.

Sampled textures first took the CPU path too. At the crash site, walking onto open sand makes the
game fast-clear a target and sample it without drawing to it, once a frame; that readback drained
the GPU behind the frame's heaviest draw (pixel shader 6996d4e234bd5b8e, up to 18.7 ms), about
25 ms a frame. A texture whose image can be a color attachment now takes the GPU path. Warm, same
route: the stretch after reaching the sand went from 16-25 fps for about 18 s to 27-39 fps for
about 11 s.

**GPU-built indexed indirect draws:** once a frame, everywhere at the crash site, the game issues
an indexed indirect draw (not mesh-emulated) whose arguments a compute dispatch wrote. Reading
them on Thread_Gpu faulted and drained the GPU: 2.6-3.5 ms a frame, and 8.5-9.6 ms on the sand,
where the drain waited behind the expensive shader. An indexed list draw with 16- or 32-bit
indices now draws from the arguments on the GPU (`vkCmdDrawIndexedIndirect`): the mesh
arguments pass, in a plain mode, copies them into the draw record ring with the count clamped to
INDEX_BUFFER_SIZE, as the CPU path clamps it (the guest layout is VkDrawIndexedIndirectCommand's).
Strips (which could need a primitive restart scan) and 8-bit indices keep the CPU path;
`--gpu-mesh-indirect false` turns both GPU paths off. Full drains dropped from one a frame to none;
on the sand, frames went from GPU and CPU in series to GPU-bound (35-48 fps, busy up to 19 ms,
gaps 6-10 ms).

**Uniform V_READFIRSTLANE (the SRT session's 7c38c759):** the lighting pixel shaders' light-list
waterfall read its key with a subgroup shuffle, which the AMD compiler lowers to ds_bpermute and
treats as divergent, so per-light loads, the key math and the list walk ran per lane. Taking the
first lane of the shuffled value makes it uniform (v_readfirstlane into an SGPR): 96 vector
buffer loads became 1, and VGPRs fell 134 to 128 (969c5ed1), 177 to 139 (5a10a907) and 140 to 120
(9d7e6bca). Warm, standing still at the start of play (two alternating rounds): 9d7e6bca 177.6 to
147.0 us/Mpx (-17%), 5a10a907 -4%, GPU busy flat (10.97 to 10.94 ms). Shaders whose SPIR-V did
not change moved by up to 6% between the sides, the noise band for this view; 6996d4e2, the sand
shader, showed no change beyond it.

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

**Relaxed readback (optional):** the read still waits when the GPU has not reached the write
yet, and at the crash site that is most frames: Astro Bot's DrawThread reads a compute-written
value at guest PC 0x90736fa9e about twice a frame, 2.5-5 ms a read at 60 fps and about 12.6 ms
on the open sand, where the frame's heaviest draw delays the dispatch (21 ms of waiting a
frame). With "Relaxed GPU readback" in the settings panel (`relaxed-readback`, off by default),
a guest read of a hot, GPU-dirty page whose download is already recorded does not wait: the
page opens read-only with its previous bytes, as memory reads on hardware return whatever is
there before the GPU writes, and the download publishes the new bytes as usual. The next GPU
write to the page closes it again. Only hot pages open, since Thread_Gpu reads some GPU-written
memory directly and relies on the fault; guest writes still fault and synchronize. Warm, sand
route, same build: tick waits fell from up to 27 ms a frame to under 1 ms in every 5 s window,
the worst sand window from 38 to 44-50 fps, and loading and cutscene windows from 24-28 to
58-60 fps; 20-110 reads a second took the previous bytes. Screenshots and the cold render check
show no difference. It does not change first-use stalls (cold whole run 23.4 s against 22.7 s,
within noise).

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

**GPU-bound, and dynamic resolution:** at about 58 fps the overworld keeps the GPU busy
16.1-16.5 ms of every frame, with under 1 ms idle, while the CPU threads wait 12-15 ms a frame for it.
GPU zones put about 1.65 ms of that in emulator work (tiler 0.77, image copies 0.39, DCC clears
0.31, buffer copies 0.16). The rest is the game's own shaders: one pixel shader,
`ef31694ed8d87754`, took about 5.3 ms. The per-frame tiler and copy cost comes from full-screen
8-byte-per-pixel surfaces (guest format 71, tile mode 27) that the game writes with GPU buffer
stores and then samples, so each one is detiled again every frame.

Astro Bot scales its render resolution: refreshed surfaces in one run ranged from 3840x2160
through 3328x1872 and 2432x1368 down to 1216x684. Each run settles in either a low mode, where
the GPU is busy about 12.2 ms a frame with 4.4 ms idle and holds a flat 60, or a high mode, where it
is busy about 16.3 ms, holds about 58 fps, and misses a vblank on 13-19% of frames. The game writes
about 305 GPU timestamps a frame (RELEASE_MEM/EOP and COPY_DATA of the clock), and the emulator
fills them with the reference clock when Thread_Gpu parses the packet. Rewriting each one with
the clock at GPU completion made things worse: busy went to 17.9 ms and fps to 52 (alternating
A/B, 60.0/58.0 vs 52.1/51.7). Completion is only known per command buffer, so a frame's
timestamps bunched together, the game measured less GPU time, and it raised the resolution.
Writing each guest timestamp with its own host GPU timestamp (`vkCmdWriteTimestamp` at that point
in the command buffer, published at completion) was worse still: 53.6 fps with 7-8 Mpx render
areas and 23% missed vblanks (alternating A/B, 4 rounds). The game budgets for its measured passes,
which the host GPU runs quickly, while the emulator's own GPU work and CPU-GPU stalls go
unmeasured. Both experiments were dropped. The parse-time values partly include those stalls,
which makes them the better default. In the high mode GPU savings turn into resolution rather
than frame rate, so compare shader changes with the zone rows' cost per million pixels, not fps.

The timestamps are EOP event 0x28: five at the start of each frame (per-frame rings), about 300
begin/end pairs 0x20 apart from the game's GPU profiler (a multi-buffered region, so a frame's
values are read a few frames later), and one at the end of the frame. `KYTY_TS_LOG=<skip>` prints
1200 of them, with flips, after skipping that many.

**Dynamic-resolution headroom (`--gpu-timestamp-scale`):** a percentage (100-200, default 100 =
off) that stretches the time since the last flip in every guest GPU timestamp, so a game that
sizes its resolution from them measures more GPU time. Each flip returns to the real clock, so
timestamps stay within a frame's stretch of the CPU clock. In the overworld (alternating runs,
45 s each):

| Scale | fps per run | Render area | Frames at 25 ms or more | p99 frame |
| --- | --- | --- | --- | --- |
| 100 | 58.1, 59.8, 59.7 | 6.23 Mpx (about 3328x1872) | 11-14% | 31-32 ms |
| 110 | 60.0, 57.1 | 3.33 Mpx, then 6.23 Mpx | 0%, then 17% | 17, then 33 ms |
| 115 | 60.0, 59.7 | 2.07 Mpx (about 1920x1080) | 0%, 0.6% | 17-19 ms |
| 125 | 60.0, 60.0, 60.0 | 2.07 Mpx | 0% | 17 ms |

At 115 and above the GPU keeps about 5 ms of each frame idle (busy about 11.5 ms), and the game
holds a steady 60. After the upstream merge ab5e6a7f (which removed per-draw diagnostic atomics,
so Thread_Gpu parses faster), 115 settles one level higher, at 2432x1368 (busy 12.3-13.0 ms). It
still holds 60 once the resolution has settled, a few seconds after loading. Anything that makes
Thread_Gpu faster shortens the parse-time timestamps, so the game raises its resolution a little. At 110 it still sometimes settles at the 3328x1872 level. The game's
resolution levels are discrete, so there is no setting between those two outcomes. For a window
around 1920x1080, a 1080p internal resolution is about native.

The setting is also in the in-game settings panel: press F2 in the game window. "Dynamic
resolution headroom" steps in 5% from Off (100%) to 150% with the mouse, the arrow keys or the
d-pad. It applies immediately and is saved to `kyty_settings.ini`. Esc, F2, circle or Close closes
the panel, and the game gets no input while it is open. After a live change, Astro Bot took about
20-25 s to step its resolution down and settle at a flat 60.

The headroom does not help on the open sand of the crash site (the warm `ship` route with relaxed
readback). The dip there lasts about 8 s at 30-45 fps per second at every scale: lowest second
31.8 fps at 115, 29.6 at 140 and 28.8 at 160, with a longer dip at 160. GPU busy stays 14-19 ms a
frame in those windows, so the game's resolution does not follow the measured time there.

**Dead end: wave32 pixel shaders.** AMD's driver compiles a Vulkan fragment shader as wave64
(`s_mov_b64 exec`), while Astro Bot's pixel shaders are wave32. The 9070 XT allows a required
subgroup size for fragment shaders, so a test build requested 32 for wave32 guest pixel shaders.
In alternating sand runs (two per side, after a warm-up run), the dip's 5 s windows were 40.6-48.2
fps against 41.6-48.8 for wave64, and the steady GPU busy was 10.6 ms either way. Not adopted for
speed; adopted later for correctness (d420fd5d): a pixel shader's EXEC, VCC and ballots cover one
guest wave, so under wave64 a wave32 shader's host lanes 32-63 read lanes 0-31's masks, and an
EXECZ or VCCZ loop exit (the waterfall loops over shadow cascades and light-list tiles) can leave
them unprocessed. The fragment stage now requests the guest's wave size. At a pinned resolution
(`--gpu-timestamp-scale 200`) GPU busy and per-shader times were unchanged within noise.

**Dead end: per-lane branches for EXEC-masked regions** (the SRT session's
`exec-region-branching`, reverted). At the same DRS level neither converting every region nor only
those with memory reads saved GPU time: sand stand-still busy 8.8-9.0 ms off vs 8.9-9.2 ms on,
overworld +1.5% (`ef31694ed8d87754`) to +5% (`92b1436c8042e396`) with every region converted. An
earlier "2x resolution" result was the game picking a different DRS level between runs; compare
GPU-cost changes at `--gpu-timestamp-scale 200` (the DRS floor) and on per-frame milliseconds,
never microseconds per megapixel across different render areas.

**Where the sand dip is:** even at the DRS floor, walking onto the open sand drops to 28-46 fps.
In that window `6996d4e234bd5b8e` draws about 1.8 times a frame at about 6.7 ms per draw (2.07
Mpx), against 0.16 ms per draw at the same area when standing still in the sand. Its loops are
waterfalls over cascade and tile keys and a walk over each tile's light list, so the likely cause
is long per-tile lists (built by the light-assignment compute `6216dac0cc17c2d4`), which the SRT
session is measuring with a list-length heat map.

**Dead end: per-lane branches for EXEC-masked regions.** The IR skips an EXEC-masked region only
when no lane is enabled, and masks each write with a select. A pass (a953539d, reverted) turned
provably safe regions in pixel shaders into branches on each lane's own EXEC bit: 44-52 of about
300 permutations changed, spirv-val clean, and on the RX 9070 XT the hot lighting shaders lost up
to 9 VGPRs and 1-4% of their instructions. At a pinned resolution (`--gpu-timestamp-scale 200`
keeps dynamic resolution at its floor; ABCCBA runs), it was not faster: sand GPU busy 8.8-9.0 ms
off against 8.9 on, and in the overworld ef31694e +1.5% and 92b1436c +5%. Converting only regions
with memory reads was within noise of off. A wave runs a region's arithmetic whenever any lane is
on, so the dropped selects did not pay for the EXEC switches. An earlier run looked like a large
win because the game's dynamic resolution settled at different levels on each side; compare
shader changes at a pinned resolution, in ms per frame. The patch is kept in the branch history.

**The walking dip is the sand trail, not its lighting.** At the resolution floor the ship route
still dips to 26-35 fps, and in that window pixel shader 6996d4e2 costs about 6.7 ms per draw
against 0.16 ms standing still. Its guest code has only waterfalls over cascade and tile keys, a
walk of each tile's light list, and a uniform loop over a game-supplied count. A loop-trip heat
map (`KYTY_DEBUG_LOOP_HEAT`) shows that 6996d4e2 draws the groove Astro drags through the sand
and that every loop there stays in the lowest band (fewer than 16 passes per wave), so the light
lists are short. With the heat map on, the driver drops all of the shader's lighting and
texture work, and the dip was unchanged (t=13-18: 25.6-32.7 fps against 26.5-33.0). The cost is
the trail's geometry or rasterization (it draws through the mesh-emulated path), not its pixel
shading.

## First-use stutter (pipeline compiles)

`--drain-stats` prints a `hitch:` line for every game frame of 50 ms or more, listing what was
recorded during it: waits, submits, readbacks, `shader-compile` (a shader lookup of 1 ms or more,
i.e. a new permutation translated) and `pipeline-create` (a pipeline lookup of 1 ms or more).
`KYTY_PERMUTATION_LOG=1` also prints why each runtime-translated permutation is new (a new
shader, a new static state and which words differ, or a new specialization and which fields
differ), each module's SPIR-V size (`spirv: id= … words=`), and each pipeline's creation time
(`pipeline: vs= ps= ms=` / `pipeline: cs= ms=`).

With empty caches, booting to the title and save menu froze the game for up to 11 s in one frame.
Of the 22.5 s of stalls, almost all was the driver (26.6.4 LLPC) creating 124 pipelines at about
190 ms each, one after another on Thread_Gpu. Our own translation of the 147 permutations took
about 2.2 s. (These runs were meant to load the crash-site save, but with cold caches the save
menu's key presses fell inside stalled frames and were never seen; see the cold-start scenario
below.) `VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT` made no difference (189 vs 194 ms per
pipeline, same GPU time), so this driver ignores it. Compile time tracks SPIR-V size (median
module 11k words, p90 61k, largest 184k; the 1.0-1.4 s pipelines hold 90k-184k-word modules), and
the same stage is compiled again in every pipeline that uses it (compiling each unique stage
once would compile 64% of the words).

### Measuring a cold start

AMD's driver keeps its own on-disk pipeline cache (`%LOCALAPPDATA%\AMD\VkCache`, `.parc` files),
one set per executable location, and it survives deleting the emulator's `_PipelineCache`. A
"cold" run from a folder that ran before is warm at the driver: the same title-screen boot took
10.8 s of graphics pipeline creation the first time and 0.65 s the second (compute pipelines:
1 ms). A cold run therefore copies the build into a new folder, and afterwards deletes the driver
cache files that run created.

The scenario starts a new game in the ship save's empty slot 2, skips the intro and the
crash-landing cutscenes (Options, Down, Cross on each; the pause menu is recognized by its blue
left-edge arc and the dark Skip item), then plays a fixed minute at the crash site: walk to the
rocks under the "!" marker, turn the camera, hop about. Every key press first waits for three
steady seconds (at least 55 frames each), since a press that starts and ends inside one stalled
frame is lost. Held-key movement still covers different ground when frames stall, so the two
sides of an A/B do not see identical content.

### Pipeline libraries

A new graphics pipeline is built from `VK_EXT_graphics_pipeline_library` parts: vertex input,
pre-rasterization shaders, fragment shader and fragment output, each cached under the state it
depends on. Only parts no earlier pipeline built are compiled, and the two shader parts compile in
parallel. The parts are then fast-linked, which averages 0.45 ms. A background thread relinks
each such pipeline with link-time optimization, and the draw path swaps that pipeline in on a
later lookup, retiring the fast-linked one once the GPU is done with it. A mesh pipeline has no
vertex input part: its mesh shader is the pre-rasterization part, and its push constant range
covers the mesh and fragment stages (with a known intermittent defect; see below). RectList
pipelines stay monolithic (their tessellation
shaders are generated per vertex and pixel shader pair, and Astro Bot draws none to test a
library form with), as do all pipelines on drivers without fast linking. The switch is
"Pipeline libraries" in the settings panel (`pipeline-libraries` in the settings file), on by
default.

Astro Bot's main geometry uses mesh shaders of 84k-103k SPIR-V words, and each one appears in
several pipelines with different pixel shaders. Built monolithically, the 13 mesh pipelines of a
cold run took 7.0 s (up to 1.24 s each), recompiling the same mesh shader every time. As library
pipelines they took 3.75 s, and the cold run's stalls dropped from about 37 s to 33 s (worst
frame 7.8 s to 6.5 s), two rounds each. But 2 of about 14 runs with mesh library pipelines
rendered the whole desert blown out to saturated yellow from the first gameplay frame (sand
255,251,62 against a normal 242,191,102), and none of the 18 runs before did. A likely cause is
a separately compiled fragment part reading a mesh output that a monolithic compile would zero,
with the garbage poisoning the game's temporal auto-exposure. Mesh library pipelines stay on by
the user's choice, with this as a known issue (see `UsesLibraries` in shaders.cpp to turn them
off). With the look-ahead below, mesh libraries on and off measured 27.1-28.6 s and 30.9-32.7 s
of cold-start stalls.

Also tried: `VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT` on the library parts (the background
link still optimizes fully) cut graphics pipeline creation only 6-8% (stalls 36.4 s to 34.7 s,
two rounds). It is not adopted: new pipelines would run unoptimized code until the relink swaps
in, which is a GPU cost not yet measured. Creating the parts without
`VK_PIPELINE_CREATE_RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT` (and so without the optimized
relink) saved nothing: matched graphics pipelines took 12.96 s against 13.36 s in one cold round
each, within the 3.5% that the compute pipelines, which the flag does not touch, moved too.

Also tried: loading bounds-checked buffer elements without a branch (the SRT session's
6820afbd). Matched pipeline creation fell 9% (graphics) and 6-10% (compute), but a warm check
standing still at the start of play measured GPU busy 10.46 to 10.51 ms per frame, with shader
5a10a907 6% and ef31694e 18% slower per megapixel: the select on the load address stops the
compiler folding constant offsets into loads. One of its four runs also rendered the scene
blown out to saturated yellow for the whole run. It was reverted pending a form that selects only
the loaded value.

### Look-ahead prefetch

Most cold-start stall time sits in loading frames that create 20-30 pipelines one after another
(boot, the title screen, scene loads). When a draw creates a pipeline, the command processor
walks the rest of the command stream without executing it: a second command processor gets a
copy of the register state, and only register writes (context, shader and user-config
registers and their indirect forms, context clear/push/pop, user-data markers) and indirect
buffer calls and chains are replayed, through the same handlers. Branches and non-type-3
packets end the walk. At each draw packet, the pipeline cache predicts the draw's pipeline from
the registers alone: it translates the shaders, derives the attachment sample count and whether
a depth attachment is bound, and takes the rasterizer state; then it queues the shader library
parts no earlier pipeline built on four worker threads. The draw finds them compiled, or waits
for them. A wrong prediction only costs a compile. Draws the draw path handles without their
shaders (metadata color modes, resolves, depth/stencil copies, unknown primitive types) are
skipped, so no translation runs that the real draw would not run. One walk covers up to 256
draws and does not run again until those draws are processed. It runs whenever pipeline
libraries are on.

Cold start, two alternating rounds, mesh pipelines monolithic on both sides: the run's stalls
fell from 36.9 and 36.3 s to 30.9 and 32.7 s, and the worst frame from 7.6 and 7.8 s to 5.5 and
5.4 s. The gameplay minute did not change (4.5-5.4 s against 4.3-6.1 s): its stalls are single
new pipelines, which a look-ahead cannot overlap. In one checked run, 150 of 209 graphics
pipelines used prefetched parts. With mesh library pipelines still on (three rounds), the same
comparison was 35.6, 35.1 and 33.7 s against 28.2, 27.1 and 28.6 s.

The walk also predicts dispatches (DISPATCH_DIRECT and DISPATCH_INDIRECT, reading the dispatch
initiator for the thread-dimension flag, and skipping zero-sized direct dispatches as the
dispatch path does): the pipeline cache translates the compute shader, makes the pipeline's
layouts, and compiles the pipeline on a worker; GetComputePipeline takes it over. A compute
pipeline miss also starts a walk, and there are six workers. On the best build (mesh libraries,
look-ahead, the branch-free storage-buffer loads), one cold round each: stalls 25.5 to 23.8 s,
worst frame 3.3 to 3.0 s, with 18 of 52 compute pipelines prefetched. The largest compute
pipelines (1.1 s, 0.83 s and 0.65 s) still compile when their dispatch arrives: each is the
first miss of its loading burst, which no look-ahead precedes, or runs on an asynchronous
compute queue, whose command stream this walk does not follow.

Two follow-ups address that. When a submission created three or more pipelines, the next
submission is walked from its first packet before it runs, so a burst's later submissions are
covered from their start; single misses during play do not qualify, so ordinary frames never
pay for a walk. And compute-queue command processors (which also run on Thread_Gpu) start walks
on a miss. One cold round: stalls 23.8 to 22.5 s, worst frame 3.0 to 2.25 s (top frames
3.0/2.85/2.8 s to 2.25/2.05/1.78 s), compute pipelines prefetched 18 to 36 of 52. The two
largest compute pipelines are now prefetched but still cost about their full compile, because
they are needed right after the walk that queues them.

Pixel shader resources use descriptor set 1 and vertex-side stages set 0. Layouts with
independent sets would let each shader part ignore the other stage's set, but on this driver any
pipeline whose layout has independent sets, even a monolithic one, lost the device within the
first frames. Binding empty sets, dropping push descriptors, and linking with link-time
optimization up front did not help. So every part uses the pipeline's full layout, and a shader
part is shared only by pipelines whose two set layouts match. Library pipelines use one push
constant range for all stages.

Cold start, same build with the setting off and on, three alternating rounds (means, with the
range):

| | Off | On |
| --- | --- | --- |
| Gameplay minute: stalls (frames of 50 ms or more) | 11.0 s (9.6-13.5) | 4.8 s (4.5-5.2) |
| Gameplay minute: stalled frames | 30 (23-44) | 14 (13-15) |
| Gameplay minute: worst frame | 2.4-2.8 s | 1.0-1.2 s |
| Whole run (boot, menus, cutscenes, gameplay): stalls | 51.8 s (50.2-55.0) | 36.8 s (36.4-37.0) |
| Pipeline creation in stalled frames, whole run | 44.9 s | 30.7 s |

With the setting off, the gameplay minute spent about 12 s between 25 and 45 fps while Astro
reached the rocks; with it on, the minute held 60 apart from one second at the start of play.

Most of what remains is driver compile time in loading frames that create 20-30 pipelines one
after another (one took 6.2 s), and compute pipelines (6 s over the run), which have no library
form.

### Asynchronous pipelines

With "Asynchronous pipelines" in the settings panel (`async-pipelines` in the settings file,
off by default), a draw whose new graphics pipeline still needs a shader library part does not
wait: the missing parts are queued ahead of look-ahead prefetches, the draw is skipped, and the
same draw in a later frame links the pipeline once its parts are compiled. Objects and effects
seen for the first time appear a few frames late instead of freezing the game. Draws whose
shaders store to buffers or write storage images are never skipped, since later work may read
what they write; buffer atomics alone do not count (Astro Bot's geometry shaders record a
per-object maximum with one). Compute dispatches always wait.

The first try left the gameplay minute at 21-36 fps for half a minute: the six compile threads
and the link thread, at normal priority, took CPU time from the game while they worked through
the burst. All background compile threads now run below normal priority. Cold start, one round
each, same build:

| | Off | On |
| --- | --- | --- |
| Gameplay minute: stalls (frames of 50 ms or more) | 4.2 s (13 frames) | 0.87 s (8 frames) |
| Gameplay minute: worst frame | 1.13 s | 0.20 s |
| Whole run: stalls | 22.6 s | 10.6 s |
| Graphics pipeline creation in stalled frames, whole run | 16.4 s | 4.3 s |

154 pipelines were deferred, skipping 5333 draws in all. What remains in play is shader
translation (50-180 ms frames, in the draw path and in look-ahead walks); in loading screens it
is compute pipelines (1.2-1.3 s each at boot and scene loads) and translation. The game registers
its shaders (`AgcCreateShader`) 7 to 128 s before their first pipeline, but a pipeline also
depends on the resource descriptors bound at the draw or dispatch, so compiling at registration
could only guess.

With the setting on, shader translation also moves off Thread_Gpu: a draw whose shaders are not
translated yet queues the translation on the worker threads (ahead of look-ahead jobs) and is
skipped, and look-ahead walks queue translations instead of running them. Whether a draw's
shaders store data is decided before translating, by decoding them (buffer, flat and image
stores, image atomics, GDS); such draws translate and wait as before. The link thread keeps
normal priority, since draws run the slower fast-linked pipeline until it relinks one: with it
below normal and the workers full of translations, GPU time per frame doubled for a while. One
cold round with everything on: gameplay-minute stalls 0.064 s (one frame of 64 ms), whole run
7.1 s, worst frame 2.1 s (boot, compute pipelines); the minute held 60 after its first 10 s
(47-56 fps while the burst compiled). 179 pipelines were deferred, 14236 draws skipped.

In this mode a look-ahead prediction advances one step per walk: the first walk queues the
translation, the next compiles the module, and only a third can queue the pipeline. The walk
covers the next draws, though, so none ran again until those were processed, and draws that
cannot be skipped (compute dispatches, draws that store data) then compiled on Thread_Gpu. The
walk now repeats while its background work finishes: at the next draw once all of it is done, or
once some of it is and 16 ms have passed. Cold loads of the ship save's slot 1 into the overworld
(`scratchpad\cold-area.ps1`), three rounds a side:

| | Before | After |
| --- | --- | --- |
| Load: stalled time | 1334, 733, 704 ms | 703, 636, 683 ms |
| Load: worst frame | 650, 383, 368 ms | 300, 250, 285 ms |
| Load: pipeline creation in stalled frames | 337-358 ms | 253-271 ms |
| Boot and title: stalled time | 5.46-5.65 s | 5.17-5.46 s |
| Look-ahead walks (time) | 116-120 (68-70 ms) | 146-154 (102-105 ms) |

The load's two compute pipelines, 58-61 ms each before, now come prefetched (3-25 ms). The rest
is a draw that stores data and whose pixel shader first appears about 30 ms before it is needed
(145 ms), and compute shaders that appear only as they are dispatched. (The first "before" run's
650 ms was Thread_Gpu waiting for the game, not pipeline work.) The overworld held 60 after every
load. Without asynchronous pipelines nothing changes: those walks translate and prefetch at once.

Also tried, without asynchronous pipelines: translating the look-ahead's shaders on the worker
threads (in up to four passes, or in one pass followed by an in-order pass that waits for each
draw's own shaders) instead of one after another on Thread_Gpu. Look-ahead time fell from about
2.9 s to 1.8 s a run, but pipeline creation grew by the same amount, since the translations
compete with the pipeline compiles for the workers and the first draws' parts start later.
Whole-run stalls stayed at 22.6-23.0 s. The walk still translates serially in that mode.

Compiling at registration (`AgcCreateShader`) would need a pipeline's specialization before any
descriptor is bound. Over a cold run's 293 permutations (`--spec-guess`), guessing every image
as a float 2D image with one mip level gives byte-identical SPIR-V for 244, including all the
largest shaders; also guessing the buffers' strides, which the SPIR-V bakes in, leaves 97, and
none of the large ones.

Hitch counts at the default GPU timestamp scale also catch the game's dynamic resolution: some
runs spend 10-15 s of play with GPU time per frame up to 19 ms (frames of 28 ms) and no pipeline
work at all. Comparisons should use `--gpu-timestamp-scale 115`, at which Astro Bot holds 60.

To see where a stalled frame's time goes, hitch lines also report `lookahead` (look-ahead walks,
mostly the shader translation they run) and `gpu-thread-idle` (Thread_Gpu waiting for the game).

## Settings file

`kyty_settings.ini` in the working directory holds default options, one per line, named like the
command-line flags without `--`; `#` starts a comment, and a switch needs no value:

```text
gpu-timestamp-scale = 115   # dynamic-resolution headroom
pipeline-libraries = false  # build every graphics pipeline in one piece
async-pipelines = true      # skip draws whose new pipeline is compiling
relaxed-readback = true     # game reads of GPU-written memory may see the previous value
fullscreen
```

It is read at startup, and the same option on the command line overrides it. The settings panel
writes its changes here.

Occlusion queries are not a culling lever here: about 8 ZPASS_DONE dumps a frame and no
predicated draws, which fits sun or lens-flare visibility checks. They still report always
visible.

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
--gpu-timestamp-scale N       Default 100 (off): stretch in-frame GPU timestamp time by N percent.
                              Astro Bot: 115 holds a steady 60 at about 1080p internal.
--drain-stats <seconds>       Report GPU waits by cause every N seconds.
KYTY_GPU_ZONES=1              With --drain-stats: also report GPU time by zone and shader hash.
KYTY_TS_LOG=<skip>            Print 1200 guest GPU timestamp writes after skipping that many.
```

`KYTY_GPU_ZONES=1` (an environment variable, diagnostic only) writes a bottom-of-pipe timestamp
wherever the recorded work changes zone: a guest draw (keyed by pixel shader hash, or vertex shader
hash without one), a predicated guest draw, a guest dispatch (keyed by compute shader hash), or
emulator work (tiler, DCC clear, fault buffer, blit, image copy, buffer copy, mesh-args pre-pass).
Texture uploads, downloads, copies and DCC clears key their emulator work by the image's guest
address (bit 63 set for downloads to guest memory). Each drain-stats interval then prints a
`gpu-zones` line with ms/frame per zone, and the 24 costliest zone keys. The shader hashes match
the shader dump names (`--graphics-debug-dump`). Overlapping work is charged to the zone whose
timestamp it finishes before, so treat the numbers as a ranking. Draw rows also give the render
area per run and the cost per million pixels of it (`area=… …us/Mpx`), which stays comparable
when the game's dynamic resolution changes. With zones on, the texture cache prints each image
the first time it is refreshed from guest memory (size, format, tiling, whether a CPU write or
a GPU buffer write caused it, and which binding refreshed it). It also prints the first GPU buffer
write of each kind (shader store, fill, copy) to land on an image: its range, whether it covers
the whole image, and whether the image held rendered content. In Astro Bot's overworld, compute
stores into buffers that alias transient render targets (for example 8.8 MB inside the 66 MB
3840x2160 surface) make the next render-target use re-upload the whole image from buffer memory.
That costs about 0.5-0.8 ms of GPU time a frame. The bytes outside the stored range come from a
buffer copy that the rendered image was never written back to. The instrumentation leaves the shader caches valid, and with a warm
cache it cost about 0.2 fps in the overworld.
The `frame-times` line gives p50/p95/p99 of the time between new game frames and the share of
frames that took 25 ms or more, i.e. missed a 60 Hz vblank. Drain stats also count occlusion
queries, occlusion predications, and GPU timestamp writes per frame.

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
| Compiled, `perf/draw-cpu` (93a8a7f9) | 1.14-1.21 |
| Compiled + memo, unchanged inputs, `perf/draw-cpu` | 0.92-0.98 |

Both paths made the same 9,972 direct reads per pass. Strict reads fell from 724-1,147
to 140-395 per pass with 64-byte blocks. The memo costs about 3% when every refresh misses, so it
pays off once more than about 12% of refreshes reuse descriptors; `KYTY_SRT_STATS=1` reports the
game's actual rate. The benchmark also reports that 305 of 308 records have no declared hash, so
`GetShaderParams` hashes their code (4.4 KB average, about 120-210 ns with XXH3) on every lookup.
It does not measure permutation search, key building, GPU execution, frame time, or Astro Bot FPS.

### Shader compile cost

```powershell
.\_Build\windows\shader_recompiler_compute_tests.exe --spirv-digest $env:TEMP\astro.shaders
.\_Build\windows\shader_recompiler_compute_tests.exe --pipeline-compile-time $env:TEMP\astro.shaders 900aba8df9448d3d 5 before.spv after.spv
.\_Build\windows\shader_recompiler_compute_tests.exe --dump-shader $env:TEMP\astro.shaders 900aba8df9448d3d dump
```

`--spirv-digest` recompiles every permutation in a journal copy, as the boot-time replay does, and
prints one line per permutation (SPIR-V XXH3 digest, words, milliseconds) and the totals. Diff the
digest columns of two builds to prove that a recompiler change leaves the SPIR-V byte-identical.
`--pipeline-compile-time` times pipeline creation for one recorded compute or pixel shader. A pixel
shader gets a generated pass-through vertex shader and 16-bit float targets. With SPIR-V files
(for example `--dump-shader` output from two builds) it times each file, interleaved round by
round; a `noopt:` prefix adds `VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT`, which the AMD
Windows driver ignores. It patches the SPIR-V generator word before every compile so the
driver's shader cache cannot serve it; the "unchanged repeat" column shows what a cache hit
costs (about 0.1 ms). Compute pipelines pin the guest wave size as the runtime does. With
`VK_KHR_pipeline_executable_properties` it also prints the driver's statistics per stage (on
AMD: used VGPRs and SGPRs, LDS, scratch). With `KYTY_PIPELINE_ISA=<dir>` it writes the driver's
LLVM IR and ISA there. `--journal-opcodes <journal> <OPCODE>` lists the recorded shaders that use
a guest opcode. Together they answer codegen questions (scalar vs vector loads, register
pressure) without a game run.

`KYTY_DEBUG_LOOP_HEAT=<hash>:<pc>[,<pc>[,<pc>]]` (hexadecimal) makes one pixel shader export, in
place of its colour targets, how many times its wave ran each guest PC (red, green, blue), in
bands: 0 black, 1-15 0.0625, 16-63 0.25, 64-255 1, 256-1023 4, 1024 or more 32. The shader's own
results stay referenced, so its resources and recorded specializations do not change, but the
driver drops their computation. Loop headers are the backward-branch targets in `--dump-shader`'s
guest disassembly.

Recompiler time for the 308 Astro Bot permutations, by pass, on the 7800X3D (September 25,
after the structurizer change): `TranslateProgram` about 616 ms, `RewriteToSsa` 427,
`EmitProgram` 416, `Structurize` 249, constant propagation 186, `BuildSrtPlan` 107,
`RemoveIdentities` 106; 2363 ms in all. The two largest compute shaders also spend 0.85-1.27 s
each in the AMD driver.

**Dead end: unoptimized first compiles.** A fast unoptimized pipeline, replaced by an optimized
one in the background, only pays if the driver compiles much faster without optimization. It
doesn't: with `VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT` (`KYTY_TEST_DISABLE_OPT=1` with
`--pipeline-compile-time`) the median compile of `900aba8df9448d3d` went from 1034 to 947 ms,
`01d6f21611218e74` from 805 to 756 ms and `5bb5e709d757909d` from 228 to 219 ms (-4% to -8%).

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
