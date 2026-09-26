# Performance roadmap (draft)

Status on September 25, 2026: research notes. Items implemented since are listed under "Done
since".
Full agent notes are in [raw-findings.md](raw-findings.md). A snapshot of upstream's open pull
requests (fetched as `refs/pull/<N>/head`) is in [upstream-open-prs.txt](upstream-open-prs.txt).

## What the profile says

Astro Bot on a Ryzen 7 7800X3D and RX 9070 XT ran at about 27 fps with the GPU about 40% busy.
The guest GPU thread (`Thread_Gpu`, `GuestGpu::ThreadRun`) was the bottleneck:

- At least 38% of its time was spent in full GPU drains for CPU readbacks.
- About 12% went to the SRT walker (`SrtWalker::EvaluateWide`, `Value::Resolve`). This was first
  read as shader translation, but the run compiled no shaders after the startup replay. It is the
  per-draw `MaterializeResources` call on every shader cache hit (see the second pass in the raw
  notes).

The CPU and GPU take turns instead of overlapping.

### Measured with `--drain-stats` (September 25)

Per frame in Astro Bot, before the DCC fix:

| Cause | Drains | ms/frame |
| --- | ---: | ---: |
| DCC fast-clear check at render-target binding (keys written by a compute shader) | ~5 | 5-7 |
| Guest-thread read fault | 1 | 1-9 (scene-dependent) |
| Thread_Gpu reading GPU-written indirect draw arguments | 1-2 | 2-3 |
| Thread_Gpu fault during a dispatch | 1 | 1-2 |

Every DCC check found a real clear. The DCC drains are now gone (GPU conditional clears, see
performance-amd.md). On the overworld the guest-thread read fault is the largest remaining
drain, at about 9.5 ms per frame.

### Done since (September 25, overworld, pixel-verified runs)

Items 1 and 3 below are implemented, along with idle-only label submits and eager readback of
hot pages. The overworld went from 25.7 to 39.1 fps; performance-amd.md has the table.
Thread_Gpu is now the saturated thread, at about 72% of a core plus its remaining read-fault
waits. The next costs on it are per-draw resource materialization (the SRT work on
`perf/srt-materialization`), per-draw shader hashing, and the mesh-emulated indirect-args
drain.

## Can we have multi-core support?

Mostly, it already exists. Guest x86-64 code runs natively, and every guest pthread is its own
host thread, so the game's own CPU work already spreads across cores. The limit is that everything
the GPU needs goes through one thread. That thread parses PM4, looks up caches, translates shaders,
records Vulkan commands, submits them, and waits on readbacks.

Ways to move work off that thread, cheapest first:

1. **Stop draining on behalf of other threads.** When a game thread faults on GPU-written memory,
   the GPU thread records the download and then blocks until the GPU is idle. Instead, it can
   record, submit, and return the tick; the faulting thread then waits for that tick itself.
   `MasterSemaphore::Wait` and `WaitPriorityOperations` are already thread-safe. This needs a
   per-page "readback armed" token so that a GPU write recorded meanwhile is not lost (see
   Buffer readback in the raw notes).
2. **Compile shaders in parallel.** Journal replay (`PipelineCache::ReplayPrecompiled`) is
   single-threaded and holds `m_mutex` through translation and `vkCreateShaderModule`. It can use
   N workers and take the lock only for dedupe and insert. Pipelines can also be recorded and
   pre-created at startup. Opt-in async compilation with skip-draw is a larger follow-up.
3. **Submit from a separate thread.** A queue-owner thread takes over `vkQueueSubmit` and
   `vkQueuePresentKHR`; `queue_mutex` is shared with present, so a slow present blocks the GPU
   thread's submits today.
4. **Let the guest build the next frame while the GPU thread works** (bounded frames in flight at
   `GuestGpu::Done`, which currently waits for the GPU thread to go idle).
5. **Large:** Yuzu-style deferred command recording on a worker thread; a second Vulkan queue for
   guest async compute. Splitting PM4 parsing from Vulkan recording was evaluated and is not
   recommended yet.

## Ranked next steps

### Quick wins (days each, measure every one against upstream on the same save and route)

- **Tag pending image downloads with their tick.** `TextureCache::InvalidateMemory`, `FreeImage`
  and `SynchronizePredicate` would wait for that tick only, off the GPU thread when it is already
  submitted, instead of `Wait(CurrentTick())`.
- **Adaptive readback window.** Download the whole dirty extent of small buffers, and grow the
  window for sequential faulters, so one wait covers a streaming read instead of one drain per
  512 KiB.
- **Coalesce fault readbacks.** Readbacks queued in one `ProcessCommands` batch share one submit.
- **Fix millisecond-granularity waits on Windows.** `Common::CondVar::WaitFor` rounds sub-ms waits
  up to 1 ms. The blocked-queue poll in `ThreadRun` sleeps 100 ms.
- **Parallel journal replay.** Uses all cores at startup.
- **Faster first-use shader compiles.** Done in part on `perf/draw-cpu`: the structurizer's
  post-dominator order (a full journal recompile 3471 to 2363 ms, byte-identical SPIR-V) and the
  EXEC-masked select-chain collapse (AMD driver compute compiles 8-15% faster on the two largest
  shaders), and branch-free storage-buffer loads (b3ffaa5f; cold graphics pipeline creation -14%,
  GPU time unchanged or better). Next: branch-free stores (LDS discard slot; storage stores under
  `robustBufferAccess2`); `TranslateProgram`, `RewriteToSsa` and `EmitProgram` are the
  largest recompiler passes (about 616, 427 and 416 ms per journal). Use `--spirv-digest` and
  `--pipeline-compile-time` (performance-amd.md).
- **Instrument the drains.** Done: `--drain-stats`.
- **Make the shader journal and driver cache survive commits.** Done: both are keyed on a hash of
  the recompiler and pipeline sources.
- **DCC fast-clear materialization without readback.** Done with conditional rendering.

### Medium (1-3 weeks)

- **Asynchronous fault readback** with per-page arm tokens, finalized in the priority callback
  (item 1 above). This is the biggest expected win for the 38%.
- **Cheaper per-draw resource materialization** (the 12%): memoize the snapshot per source entry
  when the plan is a pure function of user data and flat SRT slots, or compile the plan into a
  flat evaluation program. Also avoid the texture-cache lock on clean SRT reads.
  *Done on branch `perf/srt-materialization`:* compiled plans, a per-entry descriptor memo,
  64-byte clean-read blocks and a last-hit permutation check. See performance-amd.md for the
  benchmark; the Astro Bot effect is not yet measured.
- **Shader code hashing per draw:** 305 of the 308 recorded Astro Bot shaders have no declared
  hash, so `GetShaderParams` (shader.cpp) hashes the whole code (about 4.4 KB, 120-210 ns with
  XXH3) for every stage of every draw and dispatch. This is the ~5% `GetShaderParams` share in
  the profile. Caching the hash per registered shader address (invalidated by
  `ShaderMapUserData`) would remove it, provided games never rewrite registered code in place.
- **DCC fast-clear materialization without readback.** Remember known fill values;
  `MaterializeDccClear` currently drains whenever the metadata range is GPU-dirty.
- **Queue-owner submit thread.**
- **Bounded guest frames in flight.**
- **Fewer shader permutations** (push-data placement, clip-space constants, depth bounds).

### Large

- Opt-in async shader compilation with skip-draw.
- GPU-side indirect draws (no CPU dereference of args).
- Deferred command recording.
- Second queue for async compute.

## How to measure

Keep the game version, save, route and settings fixed, and warm each build's shader cache
separately. Compare fresh-frame times and a CPU profile of `Thread_Gpu`, and check image
correctness. For scheduler and cache changes, the GPU test groups can now run without a GPU:

```bash
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json KYTY_TEST_SOFTWARE_GPU=1 \
  ctest --test-dir _Build/linux --output-on-failure
```

On Mesa 25.2 lavapipe, 43 of 47 targets pass. `shader_recompiler_compute`,
`compute_meta_clear_classification`, `shader_precompile_gpu` and `kernel_file_system` need
features, formats, a clean build, or a video device that lavapipe in a container does not provide.

## Research status

All thirteen areas are now covered. The last eight (texture-cache lookups, DMA, fault handling,
command-processor waits, per-draw CPU cost, host sync, frame pacing, and upstream PR triage) are in
the "Second pass" section of the raw notes. Main results of that pass:

- **DMA drains:** `DmaData` has no wait of its own. The drains the profile attributed to it can
  only come from staging-ring wraps (uploads or BDA page-table writes) or from faults in its CPU
  fast paths.
- **Texture lookups:** `FindImage` → `MaterializeDccClear` is a real full-drain path for DCC
  targets whose metadata is GPU-dirty.
- **Frame pacing:** vblank and present share one thread with a 1 ms `Flip(0)` sleep and take
  `RenderContext::m_mutex`, so drains inside draws delay presents.
- **Upstream PRs:** none of the open PRs addresses the Thread_Gpu bottleneck.
