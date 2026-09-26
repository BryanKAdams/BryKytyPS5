# Raw research findings (September 25, 2026)

Automated code-reading agents produced these notes. Four of thirteen areas finished before the session
was stopped: command scheduling, buffer readback, shader/pipeline creation, and threading. The texture
cache, DMA, fault handling, command-processor waits, per-draw CPU cost, host sync, frame pacing, and
upstream-PR triage areas were covered in a second pass, appended at the end of this file. Treat every
claim as a lead to verify against the code; file:line references are for revision 791ac3a.

## Command scheduling and completion tracking (GPU-drain map)

Each CommandScheduler owns one Vulkan timeline semaphore, called MasterSemaphore. CurrentTick() is the value that the open (recording) command buffer will signal when it is submitted. Submit() does two things under GraphicContext::queue_mutex: it takes NextTick() and it calls vkQueueSubmit with a timeline signal of that value. So every tick below CurrentTick has already been submitted, in increasing order, on the single VkQueue that everything shares. A timeline signal covers all earlier submissions on that queue, so waiting for any tick also waits for everything submitted before it.

The operations:
- Flush = Submit + BeginNext.
- FlushAndWait = Submit + host wait for that tick + BeginNext.
- Finish = Submit (if a buffer is open) + host wait for CurrentTick-1 + BeginNext + PopPendingOperations.
- Wait(t): if t < CurrentTick it is a pure host wait. If t == CurrentTick it does Submit + wait + BeginNext, which is an implicit full drain.

There are two deferred-callback queues:
- Normal ops (DeferOperation) run on whichever thread calls PopPendingOperations. That is the GPU thread at the start of each draw or dispatch, in Finish, and in fault processing. An op runs only once its tick is complete (IsFree(t)) and after every priority op with tick ≤ t has finished.
- Priority ops (DeferPriorityOperation) run in FIFO order on a dedicated jthread. For each op it host-waits the op's tick and then runs it. They write downloaded bytes into guest backing with WriteBacking, and they complete flips and interrupts.

WaitPriorityOperations(t) blocks until no priority op with tick ≤ t is queued or running. So "the GPU finished tick t" and "t's downloads are published" are separate events, and every publication site waits with Wait(t) + WaitPriorityOperations(t).

The guest GPU thread (Thread_Gpu) owns the main scheduler's recording state and every cache index. Other threads reach it through GuestGpu::SendCommandSync. Called on the GPU thread, it runs the lambda inline. Called from any other thread, it queues the lambda, which the GPU thread serves between PM4 packets, and the caller blocks on a binary_semaphore until the lambda finishes. Every readback site records its download into the open command buffer and then calls Wait(CurrentTick()). That submits and drains everything, on the GPU thread, even when the requester is a game thread sitting in a page-fault handler. No resource records a last-write tick, and pending image downloads carry no tick, which is why the waits target CurrentTick.

MasterSemaphore::Wait/Refresh and WaitPriorityOperations are thread-safe, so a faulting thread could itself wait on a tick that has already been submitted. What is not thread-safe is recording and submitting, and the GPU-dirty bookkeeping (BufferCache::m_gpu_modified_ranges, which only the GPU thread may touch).

### Wait site: src/graphics/host_gpu/renderer/cache/bufferCache.cpp:247-276 BufferCache::ReadMemory (wait at 266-271: tick=CurrentTick(); m_scheduler.Wait(tick); WaitPriorityOperations(tick); UnmarkRegionAsGpuModified; then MarkRegionAsCpuModified at 272-274 when is_write)

- **Kind:** FULL DRAIN, but only when DownloadBufferMemory recorded copies. Wait(CurrentTick) takes the tick==CurrentTick branch (commandScheduler.cpp:196-204): it submits the open command buffer, host-waits it, and runs BeginNext. It then calls WaitPriorityOperations(tick) and does not call PopPendingOperations.
- **Thread:** Always Thread_Gpu. When the caller is the GPU thread, the lambda runs inline through SendCommandSync (graphicsRun.cpp:146-148). Otherwise the caller (a game thread inside the vectored exception handler, or a kernel file-I/O thread) blocks at graphicsRun.cpp:150-155 until the GPU thread runs the lambda. The GPU thread serves it either in ProcessCommands between PM4 packets (graphicsRun.cpp:697-700 -> 129-142) or in ThreadRun's command branch (491-495, 531-541).
- **Reached from:** Game CPU read of a GPU-dirty page: KytyExceptionHandler (loader/runtimeLinker.cpp:790-802) -> Memory::HandleGpuFault (kernel/memory.cpp:929-931) -> RenderContext::HandleFault (renderContext.cpp:68-69) -> BufferCache::ReadMemory(v,1,false) -> SendCommandSync -> lambda; Game CPU write to a GPU-dirty page: HandleFault (renderContext.cpp:65-66) -> BufferCache::InvalidateMemory (bufferCache.cpp:239-245) -> MemoryTracker::InvalidateRegion on_flush (memoryTracker.h:57-78) -> ReadMemory(v,1,true); Kernel file reads into GPU memory: fileSystem.cpp:162/622/747/824/850 -> Memory::InvalidateMemory (memory.cpp:917-922) -> RenderContext::InvalidateMemory (renderContext.cpp:74-80) -> BufferCache::InvalidateMemory -> ReadMemory; GPU thread touching guest memory while emulating PM4 (faults on Thread_Gpu, so the lambda runs inline): WaitRegMem *addr (graphicsRun.cpp:345), SetPredication read (881), WriteData memcpy (377), EOP label memcpy (1194, 1244), WriteReferenceClock (388), CPU fast path of CopyBuffer/FillBuffer (bufferCache.cpp:551, 527); SynchronizePredicate Download/Drain branch (graphicsRun.cpp:842-845) <- SET_PREDICATION op 3 with wait_op (878-880); Image lookup: FindImage (textureCache.cpp:1387) -> MaterializeDccClear -> ReadMemory when the DCC metadata is GPU-dirty (1210-1214); RenderContext::UnmapMemory lambda (renderContext.cpp:120), after its own Finish
- **Protects:** Before the window's tracker GPU bits are cleared (which removes page read/write protection, regionManager.h:143-147, 171-181) and before a faulting CPU store is allowed to go ahead, the CPU backing must hold every GPU-written byte of the ≤512 KiB window. Those bytes are the byte-precise ranges in m_gpu_modified_ranges; they are subtracted when the copy is recorded, at bufferCache.cpp:127. If the bytes are missing, the game reads stale data, or a late WriteBacking (priority callback, bufferCache.cpp:179-187) overwrites a CPU store the game made later. The copy is recorded at the end of the open command buffer (bufferCache.cpp:152-178), so the only tick that covers it is the open one. That is why the code waits for CurrentTick.
- **How hot:** Most likely the dominant drain. Games poll labels and read back GPU-produced data every frame. Each first CPU touch of a GPU-dirty window from any thread costs one full pipeline drain on Thread_Gpu. Several faulting threads serialize on the GPU thread's FIFO. The GPU thread's own PM4 memory accesses to GPU-dirty pages (WAIT_REG_MEM, EOP/WRITE_DATA labels, predicates) take the same path. Guess: several to dozens of calls per frame. The 512 KiB widening at bufferCache.cpp:259-264 only amortizes reads that hit the same window.

- Idea (medium): Two-phase publication for callers that are not the GPU thread. Phase 1 (SendCommandSync, on the GPU thread): run DownloadBufferMemory for the window, record {page range, T=CurrentTick()} in a publishing map owned by the GPU thread, call Flush() (submit, no wait), and return T. If the download finds nothing but an overlapping in-flight entry exists, return that entry's T instead. The faulting thread then calls GetMasterSemaphore().Wait(T) and WaitPriorityOperations(T) itself; both are thread-safe, and T < CurrentTick, so no foreign Submit happens. Phase 2 (a second, fast SendCommandSync): if !m_gpu_modified_ranges.Intersects(page-aligned window), call UnmarkRegionAsGpuModified; drop the entry; if is_write, call MarkRegionAsCpuModified. The GPU thread keeps recording during the wait.
  - Risks: Lost GPU-modified state: a GPU write recorded between phase 1 and phase 2 would be wrongly un-marked. The design prevents this by re-checking m_gpu_modified_ranges on the GPU thread, since ObtainBuffer(is_written) re-adds ranges (bufferCache.cpp:478). That requires auditing every GPU write path so that each one adds to that RangeSet. Stale data or a clobbered CPU store: prevented because phase 2 runs only after WaitPriorityOperations(T), i.e. after the WriteBacking. Spin or livelock for a second faulter on a page that is already being published: DownloadBufferMemory returns false at bufferCache.cpp:129-131, and today nothing waits or un-marks, so the instruction would re-fault in a loop. The publishing map prevents this by giving it the in-flight T. Deadlock: the Flush in phase 1 is mandatory. If the thread waited on an unsubmitted tick, a blocked guest submission waiting for this thread's label would never flush (graphicsRun.cpp:604-615). The faulting thread must not hold locks that priority callbacks take: the address-space m_mutex (memoryAddressSpace.inc:474-479) and m_pending_download_mutex. That is the same as today. It must never be called from a priority callback (guard at bufferCache.cpp:248). Cost: one extra vkQueueSubmit and a render-pass split for each fault batch. GC, Untrack and Unmap must tolerate or skip ranges that have an in-flight entry.
- Idea (large): Separate readback lane: a second CommandScheduler/MasterSemaphore, like the existing present_scheduler (swapchain.cpp:317, 354), with its own download ring, plus a last-GPU-write tick per buffer. When the region's last write tick is below main.CurrentTick(), record the copy into a small command buffer on the readback lane and submit it immediately. It lands in the queue after all previously submitted main work and before the open command buffer. Waiters, whether the GPU thread or a game thread, then wait only for work already submitted and never for the open command buffer's work.
  - Risks: The lane must not consume the main tick. The open command buffer's bookkeeping is tagged with CurrentTick: StreamBuffer::Commit (streamBuffer.cpp:291-301), Defer*Operation (commandScheduler.cpp:233, 252), CommandPool::Commit (67), FaultManager (faultManager.cpp:147) and Image::tick_accessed_last (textureCache.cpp:1384). Renumbering it would free resources early (use-after-free or device lost), hence the separate timeline. It needs a global barrier, after EndRendering, recorded into the open command buffer right after the side submit. That orders later GPU writes after the side copy (write-after-read); the open command buffer is later in submission order, so the barrier's first scope includes the side batch. The last-write tick must be conservative for shader and BDA writes; if it undercounts, the readback returns stale data. The lane's staging must be tagged with the lane's own ticks.
- Idea (medium): For readers that are the GPU thread itself (WAIT_REG_MEM at graphicsRun.cpp:345, predicate at 881), check IsRegionGpuModified before dereferencing. If the region is dirty, record a small download, Flush, and SuspendPm4 with a 'resume when IsFree(T) and priority ops ≤ T are done' condition, instead of faulting and draining. ThreadRun can serve commands and other queues in the meantime.
  - Risks: Today the retry loop for a blocked submission sleeps 100 ms (graphicsRun.cpp:505-513), so it needs a wake-up signal (m_work_available.Signal from a priority callback) or it will add latency. The resumed packet must be idempotent. Only the graphics queue is blocked, which the existing blocked-queue logic already allows (graphicsRun.cpp:500, 547-550). Other queues could run ahead, which is legal but changes timing.
- Idea (medium): Proactive asynchronous publication. At the end of a submission (RunGarbageCollector at graphicsRun.cpp:610), call DownloadBufferMemory for GPU-dirty windows that faulted recently, followed by a normal DeferOperation that un-marks the window if !m_gpu_modified_ranges.Intersects(it). PopPendingOperations already waits for priority ops ≤ t before it runs that op (commandScheduler.cpp:223). CPU reads of data that is read repeatedly then stop faulting.
  - Risks: Wasted PCIe bandwidth when the prediction is wrong. It needs the same newer-write check and the in-flight map from the first idea, for faults that land between the download and the un-mark. Page granularity: un-mark only whole tracker pages that were fully downloaded.

### Wait site: src/graphics/host_gpu/renderer/cache/textureCache.cpp:1713-1744 TextureCache::InvalidateMemory (wait at 1732-1738 inside SendCommandSync: Wait(CurrentTick)+WaitPriorityOperations)

- **Kind:** FULL DRAIN (submit open command buffer + wait everything + priority ops), repeated while HasPendingDownload(page) is true.
- **Thread:** Thread_Gpu, reached through SendCommandSync from a faulting game thread or inline from GPU-thread faults. The check and loop run on the caller thread and take m_lock (TrackingSpinLock), but m_lock is released before SendCommandSync (1731).
- **Reached from:** HandleFault(Write) (renderContext.cpp:65-67) -> TextureCache::InvalidateMemory; RenderContext::InvalidateMemory (renderContext.cpp:79) <- kernel file I/O (memory.cpp:917-921); GPU-thread CPU fast path writes into guest memory: DmaData -> CopyBuffer memcpy (bufferCache.cpp:551) or FillBuffer std::fill (527) -> write fault on Thread_Gpu -> HandleFault -> here (inline)
- **Protects:** A CPU store must not go ahead on a page while a priority WriteBacking of an image download (textureCache.cpp:1953-1964) to that page is still pending. Otherwise the late WriteBacking would overwrite the store, or invalidating the image would drop its write watcher before publication. m_pending_downloads (textureCache.h:185) holds bare GuestRanges with no tick, so the only safe target is 'everything', i.e. CurrentTick.
- **How hot:** Low by default. Pending image downloads come only from ProcessDownloadImages, which requires Config readback_linear_images (default false, emulatorConfig.h:72), and from pressure or critical GC evictions. It is hot only if the user enabled linear-image readback or the game runs near VRAM pressure. A DMA memcpy or fill landing on such a page would show up in the profile as a 'memory copy' drain.

- Idea (small): Tag each m_pending_downloads entry with its tick (m_scheduler.CurrentTick() at textureCache.cpp:1951). In InvalidateMemory, read the maximum tick of the overlapping entries under m_pending_download_mutex. If tick < CurrentTick(), wait on the calling thread with GetMasterSemaphore().Wait(tick) + WaitPriorityOperations(tick), with no SendCommandSync and no stall of the GPU thread. If tick == CurrentTick(), send a SendCommandSync that only calls Flush(), then wait on the calling thread. Keep the existing 're-check under m_lock' loop.
  - Risks: Deadlock if the thread waits while holding m_pending_download_mutex (the callback takes it at 1959) or m_lock (the GPU thread needs it). The current code already releases m_lock at 1731. It must not run inside a priority callback (guard at 1717). On the GPU thread, any tick below CurrentTick is an older, already-submitted tick, so the wait is partial and often zero. Correctness holds because the callback removes the entry only after WriteBacking (1956-1962). A new download recorded concurrently is caught by the loop.

### Wait site: src/graphics/host_gpu/renderer/cache/textureCache.cpp:2100-2209 TextureCache::CollectGarbage (wait at 2191-2199: Wait(CurrentTick)+WaitPriorityOperations, then FreeImage+erase)

- **Kind:** FULL DRAIN while holding m_lock, whenever pending_retirement is non-empty.
- **Thread:** Thread_Gpu
- **Reached from:** ThreadRun -> Process -> RunGarbageCollector (graphicsRun.cpp:610/614) -> RenderContext::RunGarbageCollector (renderContext.cpp:153) -> RunGarbageCollector -> CollectGarbage(false); Image lookup miss under VRAM pressure: draw/dispatch binding (renderDraw.cpp:1203 lock, then descriptors.cpp:542/681/916, colorRenderTarget.cpp:359, depthRenderTarget.cpp:350/376-378) or flip ResolveSurface (swapchain.cpp:344) -> FindImage -> CollectGarbage(true) (textureCache.cpp:1358-1367), at most once per GC tick
- **Protects:** Dirty victims: DownloadImageMemory's WriteBacking must finish before the image is unregistered or unprotected, because a CPU store must not race WriteBacking (comment at 2192-2194). Clean victims under pressure_only are also put into pending_retirement (2169-2172). They are erased immediately after the wait (2204-2206), so the GPU must have finished every command buffer that referenced them.
- **How hot:** Normal GC (usage ≥ trigger, about (budget-8GiB)/2) frees clean images through FreeImage -> DeferOperation (345-346) without waiting, and skips dirty ones unless pressured (2161). So it drains only at usage ≥ m_pressure_gc_memory, which for a 16 GiB budget works out to max(min(budget-4.8GiB, budget-1GiB), 1.5GiB) ≈ 11 GiB (textureCache.cpp:188-189). Likely rare on an RX 9070 XT unless VRAM is really that high. At that point every image miss can drain, once per submission. Holding m_lock during the drain makes CPU fault handlers spin in TrackingSpinLock.

- Idea (small): If the victims are clean and no download was issued, wait for max(tick_accessed_last) of the victims instead of CurrentTick. Victims touched in the current tick are already skipped (2141), so that tick is below CurrentTick: no submit of the open command buffer, and it is usually already complete. Alternatively, retire clean victims through DeferOperation, as DeleteImage does.
  - Risks: Only FindImage updates tick_accessed_last (1384). Any GPU use that skips FindImage (copies and blits, SynchronizeBufferFromImage downloads, the presentation copy, FindTexture on a cached id) would lead to vkDestroyImage while the image is still in use: device lost or corruption. It needs an audit, or a last-use tick recorded on every command recording. Deferred retirement also delays reclaiming memory right when the pressure retry needs it (1364-1367).
- Idea (medium): Asynchronous retirement. Put victims, together with the tick after their downloads, on a 'retiring' list. Keep them registered and write-protected, but have lookups treat them as misses. Finish retirement in a later pass once IsFree(tick) holds and a new non-blocking 'priority ops ≤ tick done' query returns true.
  - Risks: VRAM is released later, so there is a risk of running out of memory during pressure. A retiring image must be excluded from FindImagesInRegion lookups, or its retirement cancelled, to avoid aliasing two live images over one range. CPU faults on a retiring dirty victim still go through the HasPendingDownload path, which stays correct.

### Wait site: src/graphics/host_gpu/renderer/cache/textureCache.cpp:352-381 TextureCache::FreeImage (wait at 354-373: loop Wait(CurrentTick)+WaitPriorityOperations with m_lock released)

- **Kind:** FULL DRAIN, looping while HasPendingDownload(image range) is true.
- **Thread:** Thread_Gpu (every caller already holds m_lock)
- **Reached from:** FindImage -> lookup -> ResolveOverlap -> FreeImage (textureCache.cpp:866, 925, 929) / ExpandImage (952) / resources shrink (1352) / depth conversion (830); UnmapMemory lambda -> TextureCache::UnmapMemory (2092); CollectGarbage (2174, 2202); DeleteImage association (326)
- **Protects:** An image that overlaps a pending image download stays registered and write-watched until the download's WriteBacking completes (comment at 356-358), so CPU stores cannot race the publication.
- **How hot:** Needs a pending image download on the range, so it is rare by default (see the TextureCache::InvalidateMemory site). This is one of the 'image lookup' drain paths when downloads exist.

- Idea (small): Use tick-tagged pending downloads (same change as for TextureCache::InvalidateMemory) and wait only for the maximum overlapping download tick, which is usually below CurrentTick and already complete, followed by WaitPriorityOperations(that tick).
  - Risks: Same as the TextureCache::InvalidateMemory change. It must keep releasing m_lock during the wait (361-371) and re-fetch the slot afterward (372), because the image may have been erased meanwhile.

### Wait site: src/graphics/host_gpu/renderer/cache/textureCache.cpp:1210-1214 TextureCache::MaterializeDccClear -> BufferCache::ReadMemory (called from FindImage at 1387)

- **Kind:** FULL DRAIN, through the ReadMemory site (Wait(CurrentTick)+WaitPriorityOperations).
- **Thread:** Thread_Gpu. Inline, and usually while holding RenderContext::m_mutex (draw lock taken at renderDraw.cpp:1203/1314 before PrepareDrawRenderState), or the render lock taken during flip preparation (swapchain.cpp:725).
- **Reached from:** DrawIndex/DrawAuto (renderDraw.cpp:1189-1297) -> PrepareDrawRenderState -> color/depth targets or descriptors -> FindImage -> MaterializeDccClear; FlipQueue::Prepare (videoOut.cpp:1033) -> Presenter::PrepareFrame -> ResolveSurface -> FindImage (swapchain.cpp:344)
- **Protects:** The DCC metadata bytes must reflect GPU writes before the CPU decodes the fast-clear code and consumes it (ClearImage + FillBuffer(0xFF) at 1233-1243). Otherwise a clear is missed or a stale one is applied.
- **How hot:** Runs once per lookup of a native DCC surface whose metadata range is GPU-dirty. It is re-armed whenever the game's GPU work writes that metadata again, for example a compute fast clear or a DMA fill when the region is already GPU-dirty (FillBuffer GPU path, bufferCache.cpp:524-533). This is a plausible per-frame, per-render-target source of the 'looking up rendered images' drains.

- Idea (medium): Track provenance. When the GPU path of FillBuffer fills a whole DCC metadata slice with a constant (bufferCache.cpp:531-533), record 'known fill value' for that range, and invalidate it on any other GPU write that overlaps (ObtainBuffer is_written, CopyBuffer dst, shader or BDA writes). MaterializeDccClear then decodes the known value without a readback.
  - Risks: If any GPU write path fails to invalidate the value, the result is a wrong or missed fast clear, which is visible corruption. Partial or multi-slice fills must not qualify. BDA or storage writes from shaders must invalidate conservatively.
- Idea (large): Decode and apply the clear on the GPU: a compute pass that reads the metadata code, checks that the slice is uniform, and conditionally clears the image and resets the metadata, so the CPU never needs the bytes.
  - Risks: The clear-colour decoding (DecodeDccClear) must be reproduced in a shader. Formats and 3D views complicate it. The consumed state (FillBuffer to 0xFF) must happen on the GPU and mark the metadata as GPU-dirty.

### Wait site: src/graphics/guest_gpu/graphicsRun.cpp:823-846 CommandProcessor::SynchronizePredicate (827-828 and 839-840: BufferFlushAndWait -> FlushAndWait + WaitPriorityOperations(CurrentTick-1); 844 ReadMemory)

- **Kind:** FULL DRAIN (FlushAndWait: submit + wait + BeginNext, no PopPendingOperations) followed by WaitPriorityOperations(last submitted). The Download branch drains fully through ReadMemory.
- **Thread:** Thread_Gpu
- **Reached from:** PM4 SET_PREDICATION op 0x03 with wait_op != 0: CommandProcessor::SetPredication (graphicsRun.cpp:876-880) -> SynchronizePredicate
- **Protects:** The 64-bit predicate the CPU reads at 881 must reflect GPU producers. For image producers or pending image writebacks it takes the conservative Drain path (ClassifyPredicateSync, commandProcessor.h:21-28).
- **How hot:** Depends on how the game uses predication. Once per predicated section. Unknown for Astro Bot.

- Idea (small): In the image_gpu_modified case with no pending download, the drain does not seem to publish anything. No image download is recorded, and the CPU then reads backing that image writes never reach. It could be removed or replaced with an explicit download. In the pending-download case, wait only for the tick-tagged download tick. Alternatively, SuspendPm4 until that tick is published instead of blocking.
  - Risks: It could change behaviour in games that depend on the drain's timing. The PR #702 pending-writeback race tests must still pass. A resumed SET_PREDICATION must be idempotent.

### Wait site: src/graphics/guest_gpu/graphicsRun.cpp:1569-1571 CommandProcessor::SynchronizeGpu -> CommandScheduler::Finish (caller 1217)

- **Kind:** FULL DRAIN (Submit, wait CurrentTick-1, BeginNext, PopPendingOperations). No priority wait.
- **Thread:** Thread_Gpu
- **Reached from:** PM4 EOP/RELEASE_MEM with event_write_source 0x01, eop_event_type 0x2f, cache_action 0, event_index 6 (GDS read-back): WriteAtEndOfPipe (graphicsRun.cpp:1212-1226) -> SynchronizeGpu -> then Sync::ReadGds reads the host-visible GDS buffer on the CPU
- **Protects:** GDS contents (m_gds_buffer, Stream memory, bufferCache.cpp:194) must include every GPU write before the CPU copies them into the label at record time.
- **How hot:** Once per GDS end-of-pipe readback packet. The frequency in Astro Bot is unknown; GDS counters, often used for append/consume, could be per frame.

- Idea (medium): Record a GPU copy from the GDS buffer to the label's cached buffer (ObtainBuffer(dst,is_written), so the label becomes GPU-dirty) instead of a CPU copy after Finish. Defer the interrupt through DeferPriorityOperation so it fires after completion. The CPU only pays when the guest actually reads the label, and then only through the publication path.
  - Risks: The label becomes GPU-dirty, so a later CPU poll takes a readback fault (and needs the ReadMemory changes to be cheap). The ordering of the interrupt relative to the label changes; this is closer to hardware but still a behaviour change. Guest code that reads the label immediately on the same queue must still see ordered data, via WAIT_REG_MEM through the same buffer cache.

### Wait site: src/graphics/host_gpu/renderer/renderContext.cpp:97-132 RenderContext::UnmapMemory (lambda 114-124: Finish + WaitPriorityOperations(tick))

- **Kind:** FULL DRAIN plus priority wait (and possibly a second, cheap ReadMemory drain at line 120 for GPU-dirty pages).
- **Thread:** Thread_Gpu, through SendCommandSync from the game thread that unmaps. When m_gpu is null or IsStopping(), it runs on the caller's thread (127-129); see facts about a race there.
- **Reached from:** Guest munmap / direct-memory unmap and remap: kernel memory.cpp:2606 (and 2321, 2333, 2413, 2876) -> UnmapGpuRange (memory.cpp:92-97) -> RenderContext::UnmapMemory, only if the range intersects m_mapped_ranges (104-107, PR #483)
- **Protects:** Before the backing goes away: no in-flight GPU access to the range, GPU-dirty data published, and pending image writebacks done. Only then are the cache entries invalidated and freed.
- **How hot:** Only if the game unmaps or remaps GPU-mapped memory during gameplay (streaming). PR #483 already removed the drain for memory the GPU never mapped.

- Idea (small): Drain only when the range intersects a registered buffer (IsRegionRegistered), an image (FindImagesInRegion), a pending download, or GPU-dirty pages. Otherwise drop the ranges without waiting. With per-resource last-use ticks, wait only for those ticks.
  - Risks: The GPU reaches guest memory only through cached Buffers and Images, including through the BDA page table. Upload staging is copied at record time. If an access path is missed (for example a BDA page-table entry that is still live), the GPU could access unmapped or reused memory. Other aliases of the same direct memory still need publication of GPU-dirty data, so do not skip the wait when GPU-dirty pages intersect.

### Wait site: src/graphics/host_gpu/renderer/cache/bufferCache.cpp:595-647 BufferCache::RunGarbageCollector (wait at 632-635: Wait(CurrentTick)+WaitPriorityOperations)

- **Kind:** FULL DRAIN, only when aggressive (critical memory) and dirty buffers were downloaded.
- **Thread:** Thread_Gpu
- **Reached from:** ThreadRun -> Process -> RenderContext::RunGarbageCollector (graphicsRun.cpp:610/614, renderContext.cpp:154)
- **Protects:** The downloads of evicted GPU-dirty buffers must be written back before their tracked pages are untracked and the owners are unregistered and erased (636-646).
- **How hot:** Rare. It needs usage ≥ m_critical_gc_memory (about budget-1.6 GiB at a 16 GiB budget; constructor at bufferCache.cpp:214-222).

- Idea (medium): Retire asynchronously: keep dirty victims registered and protected, record their tick, and finish them in a later pass when IsFree(tick) holds and the priority ops are done, instead of waiting.
  - Risks: Memory is released later, right at critical pressure. Victims must not be picked again, and ReadMemory or CPU faults on them must wait for the in-flight tick.

### Wait site: src/graphics/host_gpu/renderer/cache/streamBuffer.cpp:313-333 StreamBuffer::WaitPendingOperations (Wait(watch.tick) at 325; WaitPriorityOperations for Download usage at 326-328), called from Map 239-283

- **Kind:** Waits for a specific older tick when the ring wraps. It becomes a FULL DRAIN only if the watch being overwritten belongs to the open tick, i.e. the whole ring was consumed within one open command buffer.
- **Thread:** Thread_Gpu (these rings belong to the main scheduler)
- **Reached from:** UploadCopies -> m_staging_buffer.Map (bufferCache.cpp:427, 512 MiB); DownloadBufferMemory -> m_download_buffer.Map (bufferCache.cpp:133, 64 MiB); DownloadImageMemory -> Download ring (textureCache.cpp:1913-1915); Tiler params and draw constants -> m_stream_buffer (tiler.cpp:228, renderDraw.cpp:945, descriptors.cpp:720, 64 MiB); WriteDataBuffer -> StreamBuffer::Copy (bufferCache.cpp:33-45)
- **Protects:** The GPU must no longer read a ring region (and, for downloads, its priority WriteBacking must be finished) before the CPU overwrites it.
- **How hot:** Usually a no-op or an already-completed older tick. A drain only on very large in-tick usage. A possible hidden contributor to the drains attributed to DMA and uploads.

- Idea (small): When a Map would have to wait on a watch whose tick equals CurrentTick, Flush early (submit the open command buffer) at a safe point, or at half-ring usage within one tick, so the wrap waits only on submitted ticks. Alternatively, grow the ring.
  - Risks: More submits. Flushing mid-draw ends dynamic rendering (context.cpp:43-44) and drops bound state; the current wrap path already does this (commandScheduler.cpp:198-204), so callers are already built for it.

### Wait site: src/graphics/host_gpu/renderer/cache/faultManager.cpp:77-81 FaultManager::ProcessFaultBuffer (Wait(m_fault_areas[area]) + PopPendingOperations)

- **Kind:** Waits for a specific older tick from 8 rounds earlier (MaxPendingFaults=8, faultManager.h:15). Not a drain in practice. It also runs normal deferred operations.
- **Thread:** Thread_Gpu
- **Reached from:** ThreadRun -> Process -> RenderContext::RunGarbageCollector (renderContext.cpp:148-151) when PrepareBda set m_fault_process_pending (renderContext.cpp:144)
- **Protects:** The download area of the fault buffer must not be reused before the previous readback of that area (the deferred FindBuffer callback, 131-145) has run.
- **How hot:** Once per submission that used BDA. It almost never blocks.

- Idea (small): No change needed. Optionally replace the Wait with an IsFree check and skip this round if the area is busy.
  - Risks: Skipping loses fault reports for that submission, so buffers are created later and the shader keeps hitting unbacked BDA pages until the next round.

### Wait site: src/graphics/host_gpu/renderer/commandScheduler.cpp:210-226 PopPendingOperations (WaitPriorityOperations(op.tick) at 223) and 268-295 PriorityOperationsThread (m_master.Wait(op.tick) at 284)

- **Kind:** Not a drain. PopPendingOperations waits only for priority callbacks with tick ≤ an already-completed tick. The priority runner host-waits each op's tick on its own thread.
- **Thread:** PopPendingOperations: Thread_Gpu (renderDraw.cpp:1195/1306, renderCompute.cpp:202/408, Finish at 191, faultManager.cpp:80), plus the shutdown owner. Runner: the scheduler's m_priority_thread.
- **Reached from:** Start of every draw and dispatch (renderDraw.cpp:1195, 1306; renderCompute.cpp:202, 408); CommandScheduler::Finish (commandScheduler.cpp:191); FaultManager::ProcessFaultBuffer (faultManager.cpp:80)
- **Protects:** Normal deferred ops for tick t (for example m_slot_buffers.erase, or destroying a tiler scratch buffer) run only after the GPU finished t and after every priority callback ≤ t finished.
- **How hot:** Called for every draw. It blocks only while a large WriteBacking memcpy for an older tick is running on the priority thread, which can be milliseconds for big image downloads.

- Idea (small): Leave as is. Optionally, skip the pop instead of blocking when the priority runner is busy with an op ≤ tick (use a try variant).
  - Risks: Resource destruction is delayed, and memory is held longer.

### Wait site: src/graphics/presentation/window/swapchain.cpp:157 FramePool::WaitForFrame (present_scheduler.Wait(frame.present_tick)), via Acquire 86-108 / AcquireLast 110-130

- **Kind:** Waits for a specific older tick on the separate presentation timeline. Because the queue is shared, that tick completes only after all main work submitted before it.
- **Thread:** Thread_Gpu (PrepareFrame from FlipQueue::Prepare, videoOut.cpp:1033, and PrepareBlankFrame with a producer) and the VideoOut PresentThread (videoOut.cpp:824-866)
- **Reached from:** GPU flip: WriteAtEndOfPipe flip (graphicsRun.cpp:1545) -> PrepareVideoOutFlip (sync.cpp:179-197) -> SubmitFlipFromGpu -> FlipQueue::Prepare -> Presenter::PrepareFrame -> FramePool::Acquire; CPU flip: PrepareCpuFlip (graphicsRun.cpp:1553-1567) -> VideoOut::PrepareFlip -> same
- **Protects:** A prepared-frame image must not be overwritten while its previous present copy is still in flight.
- **How hot:** Once per flip. It normally returns immediately with N = swapchain image count frames (swapchain.cpp:320).

- Idea (small): No change needed.
  - Risks: n/a

### Wait site: Shutdown-only waits: commandScheduler.cpp:125-130 (Shutdown: Submit, Wait(CurrentTick-1), Pop, DrainPriorityOperations); renderContext.cpp:40-43 (ShutdownGpu: Finish + DrainPriorityOperations); graphicsRun.cpp:525 -> BufferWait 275-278 -> Finish; swapchain.cpp:44 (FramePool dtor); descriptorHeap.cpp:28-31 (dtor)

- **Kind:** FULL DRAIN, at teardown only.
- **Thread:** GPU thread at exit (graphicsRun.cpp:525). The main thread after the GPU thread has joined (renderContext.cpp:34-45). The presentation owner for the present_scheduler.
- **Reached from:** RenderContext::~RenderContext (renderContext.cpp:23-26); GuestGpu::Shutdown -> ThreadRun should_stop (graphicsRun.cpp:487-528)
- **Protects:** All GPU work and callbacks finish before objects are destroyed.
- **How hot:** Never during gameplay.

- Idea (small): None.
  - Risks: n/a

### Facts

- Tick semantics: MasterSemaphore starts with m_current_tick=1 and m_gpu_tick=0 on a timeline semaphore with initial value 0 (masterSemaphore.h:37-38, masterSemaphore.cpp:8-18). NextTick() is fetch_add(1, release) and returns the old value (masterSemaphore.h:26-28), so Submit signals the tick that CurrentTick() named. Afterwards CurrentTick() = submitted+1. Every tick < CurrentTick() has been passed to vkQueueSubmit. IsFree(t) ⇔ KnownGpuTick() ≥ t (masterSemaphore.h:25).
- Submit (commandScheduler.cpp:346-392) holds GraphicContext::queue_mutex (graphicContext.h:44) across NextTick() and vkQueueSubmit (359-380). Ticks are therefore signaled in increasing order on the one VkQueue. Nothing asserts which thread calls Submit. It is owner-only by convention: m_command and m_command_pool are unsynchronized, and a Vulkan command pool must be externally synchronized.
- queue_mutex is shared with presentation and tools: vkQueuePresentKHR (swapchain.cpp:696-699), Swapchain::Destroy waitIdle (502-503), ImGui overlay recording (systemOverlay.cpp:1009-1012), RenderDoc capture (renderDoc.cpp:172-173). A slow present therefore blocks every main-scheduler Submit, including the Submit inside a drain. MasterSemaphore::Wait does not take queue_mutex.
- There are two CommandScheduler instances. RenderContext::m_command_scheduler (renderContext.h:72) is recorded by Thread_Gpu. Presenter::Impl::present_scheduler (swapchain.cpp:317, 354) is recorded and submitted by the VideoOut PresentThread under RenderContext::m_mutex (swapchain.cpp:748-758, 798-803), and Thread_Gpu waits on it in FramePool::Acquire. Each has its own timeline semaphore, command pool and priority jthread (constructor at commandScheduler.cpp:96-99). present_scheduler is never Begin()'d, so Active() is false and Defer*Operation / Wait(CurrentTick) on it would EXIT (CheckActive, commandScheduler.cpp:330-332).
- MasterSemaphore::Wait/Refresh/IsFree/KnownGpuTick/CurrentTick are thread-safe (masterSemaphore.cpp:26-55). They use only atomics (acquire loads, and a CAS loop that only moves the counter forward) plus vkGetSemaphoreCounterValue and vkWaitSemaphores, which have no externally synchronized parameters. Host-waiting on a value that has not been submitted is legal, but it blocks until the owner submits. A foreign thread must only wait on t < CurrentTick() it has observed: a blocked guest submission that makes no progress is not flushed (graphicsRun.cpp:604-615), and that can deadlock with a game thread whose label write the GPU is waiting for.
- CommandScheduler::Wait(t) (commandScheduler.cpp:194-208): EXIT if t > CurrentTick. For t == CurrentTick it runs CheckActive, Submit (EXIT unless the returned tick == t), m_master.Wait and BeginNext: an implicit full drain that only the owner may do. For t < CurrentTick it is a pure host wait that any thread may call. It never pops deferred ops and never waits for priority ops. FlushAndWait (178-182) does the same as the t == CurrentTick case. Finish (184-192) submits only if a command buffer is open, waits CurrentTick-1, runs BeginNext, then PopPendingOperations, but still does not wait for priority ops, which is why callers pair it with WaitPriorityOperations.
- DeferOperation (commandScheduler.cpp:228-245) queues {cb, CurrentTick()} in m_pending_operations. It runs only when some thread calls PopPendingOperations (210-226) and IsFree(tick) holds, on that caller's thread, after WaitPriorityOperations(tick). The callers are Thread_Gpu at draw and dispatch start (renderDraw.cpp:1195, 1306; renderCompute.cpp:202, 408), Finish (191), faultManager.cpp:80, and Shutdown (129). DeferPriorityOperation (247-266) queues to m_priority_operations and notifies the runner. PriorityOperationsThread (268-295) pops in FIFO order, sets m_priority_active and m_priority_active_tick under m_operation_mutex, host-waits the tick, runs the callback with g_deferred_callback_scheduler=this, then clears the active flag and calls notify_all. Pushes come from a single producer (Thread_Gpu, with CheckActive), so the queue's ticks never decrease.
- Ordering between tick completion and callbacks. A callback for tick t never starts before the GPU has completed t (commandScheduler.cpp:284-286). Callbacks run one at a time in tick order. Completion of t does not imply that t's callbacks have run. WaitPriorityOperations(t) (304-313) returns once no queued or active op has tick ≤ t. That is complete only if t was already submitted by the time the waiter observes it: the pushes happen-before the release store in NextTick, which the waiter's acquire load of CurrentTick reads. Normal ops for t run after priority ops ≤ t (223).
- WaitPriorityOperations and DrainPriorityOperations EXIT if called from inside any deferred callback of the same scheduler, including normal ops that PopPendingOperations runs on Thread_Gpu, because RunOperation sets the thread-local for both kinds (commandScheduler.cpp:13, 298, 305, 315-320). If Thread_Gpu calls them while an op for the open tick is queued, it deadlocks: the runner waits for a submit that the blocked thread would perform. In-tree callers always submit first.
- Guards against re-entry from callbacks: BufferCache::ReadMemory (bufferCache.cpp:248-252) and TextureCache::InvalidateMemory (textureCache.cpp:1717-1721) EXIT when !IsGpuThread() && InDeferredOperation(). UnmapMemory EXITs in any deferred callback (renderContext.cpp:109-113). They exist because the GPU thread may be waiting on that very priority callback.
- GuestGpu::SendCommandSync (graphicsRun.cpp:144-156): on Thread_Gpu (thread_local g_gpu_thread, set at 472) it runs the lambda inline. That can happen inside a fault handler in the middle of a memcpy, with an open command buffer and a PM4 execution scope active. On any other thread it wraps the lambda with a std::binary_semaphore, SendCommand pushes it to m_commands under m_queue_mutex (116-127, EXIT if !m_accepting), and the caller blocks in done.acquire() until the lambda has finished. Commands take priority over guest submissions in ThreadRun (491-495). They are also drained between every PM4 packet (ProcessPm4 -> ProcessCommands, 697-700, 129-142), so service latency is bounded by one packet, which may include a runtime shader translation. RenderContext::m_mutex is not held there.
- The draw and dispatch paths hold RenderContext::m_mutex (renderContext.h:70) from renderDraw.cpp:1203/1314 and renderCompute.cpp:221/412 through binding resolution. EmitGlobalBarrier (graphicsRun.cpp:1398) and Presenter::PrepareFrame/Present (swapchain.cpp:725, 748, 798) also take it. Any drain reached from FindImage inside a draw therefore also stalls PresentThread's present submissions. Lock order in the tree: RenderContext::m_mutex before queue_mutex (renderDoc.cpp:172-173; swapchain.cpp:798-803 -> Submit).
- Texture-cache locks. m_lock is a TrackingSpinLock (textureCache.h:173): waiters spin. Priority callbacks take only m_pending_download_mutex (textureCache.cpp:1957-1962). FreeImage and InvalidateMemory release m_lock before they wait (361-371, 1729-1731). CollectGarbage keeps m_lock across its full drain (2101, 2192-2199), so a CPU fault handler that enters TextureCache::InvalidateMemory spins for the whole drain.
- MemoryTracker/RegionManager: per-region TrackingSpinLock (regionManager.h:31-49, 168). A GPU-dirty page loses both read and write access (UpdateProtection<enable,true>, regionManager.h:143-147, 172-181). CPU-dirty and GPU-dirty states are mutually exclusive and EXIT on conflict (regionManager.h:122-131). ForEachUploadRange(is_written) keeps the region lock from the CPU-dirty upload through setting GPU-dirty (memoryTracker.h:103-129). Page-protection changes already run on game threads (InvalidateRegion on the fault path, memoryTracker.h:62-76).
- BufferCache::m_gpu_modified_ranges (bufferCache.h:129) is a byte-precise RangeSet without synchronization and is owned by the GPU thread (comment at bufferCache.h:65). DownloadBufferMemory subtracts ranges when it records the copy (bufferCache.cpp:127). ObtainBuffer(is_written) adds them back (478). Tracker GPU bits are cleared only after publication (270, 638).
- No per-resource last-write tick exists. The only tick-tagged structures are Image::tick_accessed_last (image.h:151), which is an access tick set only in FindImage (textureCache.cpp:1384) and used for overlap deletion (842-844) and pressure GC (2141); StreamBuffer::Watch.tick (streamBuffer.h:110-113); the CommandPool per-command-buffer ticks (commandScheduler.h:72, .cpp:62-90); FaultManager::m_fault_areas (faultManager.h:31); DescriptorHeap pending pools (descriptorHeap.cpp:46); PendingOperation.tick (commandScheduler.h:78-81); and Presenter::Frame::present_tick (swapchain.cpp:29). TextureCache::m_pending_downloads holds bare GuestRanges (textureCache.h:185) and BufferCache GPU-dirty state is untimed.
- Publication path. DownloadBufferMemory records a barrier to eTransferRead, a copyBuffer into Download memory, and a barrier TransferWrite->HostRead with the eHost stage (bufferCache.cpp:152-178). The priority callback calls Invalidate (non-coherent memory, streamBuffer.cpp:132-133) and then Memory::WriteBacking (bufferCache.cpp:179-187). WriteBacking writes the backing alias under the address-space m_mutex (memoryAddressSpace.inc:169-177, 474-479) and bypasses page protection. Memory types: Download is host-visible random-access, preferring host memory (streamBuffer.cpp:26-27, 38). Stream and Upload are mapped sequential-write, with Stream preferring device memory (22-25, 36-37), and the GDS buffer is Stream memory (bufferCache.cpp:194).
- EOP and label semantics. Guest end-of-pipe labels are written by the CPU when the packet is recorded (graphicsRun.cpp:1191-1194, 1242-1244). Only interrupts and flip completion are deferred to priority ops (sync.cpp:199-256). A label that lands on a GPU-dirty page therefore faults on Thread_Gpu and takes the ReadMemory full drain.
- DmaData (graphicsRun.cpp:397-450), CopyBuffer and FillBuffer (bufferCache.cpp:509-567) contain no direct Wait. Drains under 'memory copies' must come through (a) CPU fast-path memcpy or std::fill faulting on Thread_Gpu (bufferCache.cpp:551, 527) -> HandleFault -> the TextureCache pending-download drain, (b) StreamBuffer wrap waits in UploadCopies, tiler or download rings, or (c) later CPU reads of a DMA destination that the GPU path made GPU-dirty (ReadMemory).
- Texture pressure thresholds are budget-based (textureCache.cpp:183-193): with a 16 GiB budget, pressure ≈ 11.2 GiB, critical ≈ 14.4 GiB and trigger ≈ 4 GiB. Below pressure, normal GC frees only clean images through DeferOperation and never drains (2157-2175). ReadbackLinearImages defaults to false (emulatorConfig.h:72), so pending image downloads are uncommon by default.
- Latent race: RenderContext::UnmapMemory calls unmap(), and therefore m_command_scheduler.Finish(), on the calling game thread when m_gpu->IsStopping() (renderContext.cpp:125-129). ThreadRun may still be recording the remaining queued submissions at that point (graphicsRun.cpp:482-523), so this is a data race on m_command at shutdown.
- Non-Vulkan stall on Thread_Gpu: PrepareVideoOutFlip loops on WaitForSubmitSlot while the flip queue is full (sync.cpp:181-196). The GPU thread then waits for PresentThread, which presents at vblank pace.
- GuestGpu::Done from a game thread calls WaitForIdle (graphicsRun.cpp:199-206, 461-466). That only waits for the guest-GPU thread's queues to become empty and does not wait for Vulkan completion.

### Open questions

- Which inner call produced the DMA-attributed drains? DmaData itself has no Wait. Stacks below DmaData are needed to tell apart fault-in-memcpy, a StreamBuffer wrap, and a later ReadMemory.
- How are ReadMemory drains split between game-thread faults (via SendCommandSync) and faults from the GPU thread's own PM4 memory accesses (WAIT_REG_MEM at graphicsRun.cpp:345, EOP/WRITE_DATA labels, predicates, MaterializeDccClear)? The first group benefits from moving the wait to the faulting thread; the second needs suspension or partial waits.
- Does Astro Bot re-dirty DCC metadata every frame, so that MaterializeDccClear drains on every lookup? Which writer does it: a compute fast clear, or DMA fill or copy on an already GPU-dirty range?
- What is GetTotalMemoryBudget and the actual VRAM usage on the RX 9070 XT, i.e. does usage ever reach the ~11 GiB pressure threshold where FindImage misses trigger CollectGarbage(true) drains? Is readback_linear_images enabled in the user's config?
- Is every GPU write path (SynchronizeBufferFromImage, JoinOverlap CopyFrom, shader storage writes, BDA writes through the page table) reflected in both the tracker GPU bits and m_gpu_modified_ranges? Moving the un-mark step of the two-phase design out of line depends on that invariant.
- Does vkQueuePresentKHR on the AMD Windows driver block long enough while holding queue_mutex (swapchain.cpp:697-698) to delay the GPU thread's Submit calls?
- How often does the game unmap GPU-mapped memory, or read GDS through end-of-pipe (graphicsRun.cpp:1217), during gameplay?

## Buffer readback (GPU-drain map)

How readback works now. GPU ownership is tracked at two granularities. MemoryTracker keeps per-4 KiB-page bits (RegionManager m_cpu_dirty/m_gpu_dirty, each 4 MiB region under its own TrackingSpinLock). BufferCache::m_gpu_modified_ranges is a plain byte RangeSet used only by the GPU thread. The only producer of GPU-dirty state is ObtainBuffer(is_written) on Thread_Gpu, at record time. SynchronizeBuffer -> ForEachUploadRange uploads CPU-dirty pages, then, still holding the region locks, sets GPU-dirty over the whole range; afterwards the bytes are added to the RangeSet. UpdateProtection gives GPU-dirty pages a PageManager access watcher (NoAccess); clean pages keep a write watcher (Read-only).

A guest access then faults on the faulting thread (KytyExceptionHandler -> HandleGpuFault -> RenderContext::HandleFault). Reads call ReadMemory directly. Writes go through InvalidateRegion, which checks GPU-dirty and marks CPU-dirty atomically under the region lock, or calls ReadMemory(is_write). ReadMemory ships a closure to Thread_Gpu with SendCommandSync; it runs between PM4 packets. The closure widens the range to a clamped 512 KiB window. DownloadBufferMemory copies only the dirty bytes into the 64 MiB host-cached download ring, subtracts them from the RangeSet, and queues a priority callback that WriteBacking()s them through the backing alias, which ignores guest protection. The GPU thread then does Wait(CurrentTick) (submit plus a wait for everything recorded) and WaitPriorityOperations. Finally it clears GPU-dirty on the window (NoAccess -> Read-only) and, for writes, marks CPU-dirty (RW). This is correct only because no recording happens between Subtract and Unmark.

Can the wait move off the GPU thread? Yes, provided that:
1. Each scheduled page gets a per-page arm token, set and cleared under the region lock. Any new GPU-dirty marking clears it.
2. The priority callback itself finalizes, after WriteBacking, the pages that still carry its token.
3. The GPU thread only records, Flush()es and returns the tick. The faulting thread waits with MasterSemaphore::Wait plus WaitPriorityOperations, then retries.
4. Every GPU-thread check that reads "no bytes in m_gpu_modified_ranges" as "the backing is current" also counts armed pages as dirty. These are TryReadGpuCleanBacking, SafeToDownload, and the image-insert check.
5. Synchronous GPU-thread callers, GC and Unmap wait for armed ticks even when nothing new is recorded.

Cached buffers use VMA AUTO_PREFER_DEVICE with no host-access or MAPPED flags. They are DEVICE_LOCAL and never CPU-mapped, so ReBAR would not remove the wait.

### Wait site: src/graphics/host_gpu/renderer/cache/bufferCache.cpp:266-271 BufferCache::ReadMemory (closure queued at :253, window :259-264, CPU mark :272-274)

- **Kind:** Full drain. DownloadBufferMemory records into the open command buffer. m_scheduler.Wait(CurrentTick()) then takes the tick==CurrentTick branch (commandScheduler.cpp:196-204): Submit the open command buffer, MasterSemaphore::Wait (vkWaitSemaphores, masterSemaphore.cpp:38-55), BeginNext. Next, WaitPriorityOperations(tick) (commandScheduler.cpp:304-313) waits until the priority thread has run the WriteBacking callback (bufferCache.cpp:179-187) and every other priority op up to that tick (image writebacks, EOP interrupts, flip completion). Then UnmarkRegionAsGpuModified(window) and, for writes, MarkRegionAsCpuModified(vaddr,size). The wait is skipped only when DownloadBufferMemory finds no dirty bytes. Separately, the non-GPU caller is blocked on a std::binary_semaphore inside GuestGpu::SendCommandSync (graphicsRun.cpp:144-156) for the whole duration.
- **Thread:** Closure: Thread_Gpu (GuestGpu::ThreadRun, graphicsRun.cpp:468). It runs from ProcessCommands between PM4 packets (graphicsRun.cpp:697-700 -> 129-142) or from the idle command branch (graphicsRun.cpp:531-541). It runs inline when the caller is already the GPU thread (graphicsRun.cpp:146-149). Publication runs on the CommandScheduler priority thread (commandScheduler.cpp:268-295). Caller: the faulting guest thread, or a kernel-syscall thread, blocks in SendCommandSync.
- **Reached from:** Guest READ of a GPU-dirty (NoAccess) page: host exception -> KytyExceptionHandler (src/loader/runtimeLinker.cpp:782-802) -> LibKernel::Memory::HandleGpuFault (src/kernel/memory.cpp:929-931) -> RenderContext::HandleFault read branch (renderContext.cpp:68-69, size 1) -> BufferCache::ReadMemory(addr,1,false) -> SendCommandSync -> Thread_Gpu closure; Guest WRITE to a GPU-dirty page: ... RenderContext::HandleFault write branch (renderContext.cpp:65-67) -> BufferCache::InvalidateMemory (bufferCache.cpp:239-245) -> MemoryTracker::InvalidateRegion (memoryTracker.h:57-78). GPU-dirty is checked under the region lock at :67-72, and on_flush runs after unlock at :74-76 -> ReadMemory(v,s,true); Kernel I/O about to write guest buffers (ReadDirectory/read/pread/readv, src/kernel/fileSystem.cpp:162,622,747,824,850) -> Memory::InvalidateMemory (memory.cpp:917-922) -> RenderContext::InvalidateMemory (renderContext.cpp:74-81) -> BufferCache::InvalidateMemory -> ReadMemory(is_write). The range must be CPU-dirty (RW) on return because ReadFile/recv into a protected page does not raise a user-mode fault; Thread_Gpu inline: TextureCache::FindImage (textureCache.cpp:1387) -> MaterializeDccClear (textureCache.cpp:1179) -> ReadMemory(dcc range) when the DCC metadata is GPU-dirty (textureCache.cpp:1211-1214); Thread_Gpu inline: PM4 SET_PREDICATION op 0x03 with wait_op (graphicsRun.cpp:876-881) -> SynchronizePredicate (graphicsRun.cpp:823-846) -> ReadMemory (:844); Thread_Gpu self-fault: CPU-side dereference of GPU-written guest memory during PM4 emulation. Examples: DrawIndirect/DrawIndexedIndirect args (graphicsRun.cpp:1019,1029), DispatchIndirect with thread dimensions (graphicsRun.cpp:1120), eager EOP label memcpy (graphicsRun.cpp:1194), WriteReferenceClock memcpy (graphicsRun.cpp:388). Each goes HandleFault on the GPU thread -> SendCommandSync runs inline -> synchronous drain; Thread_Gpu: RenderContext::UnmapMemory closure (renderContext.cpp:114-124), after Finish+WaitPriorityOperations -> BufferCache::InvalidateMemory -> ReadMemory(is_write)
- **Protects:** (1) The guest never observes a page before the GPU-written bytes reach the backing. GPU-dirty pages hold an access watcher (NoAccess, pageManager.cpp:88-96) until Unmark (:270). (2) WriteBacking writes through the backing alias (memoryAddressSpace.inc:169-178, 474-500), which ignores guest protection. It must finish before the page is unprotected, or a CPU store made after unprotection would be overwritten by the late publication. (3) Byte and page ownership are re-synchronized. Bytes leave m_gpu_modified_ranges at record time (bufferCache.cpp:127), but the page bits clear only at :270. The two sets are equal only because nothing is recorded on Thread_Gpu in between, so a re-dirty cannot be lost. (4) A write fault ends CPU-dirty only after the page is no longer GPU-dirty; ChangeState EXITs on CPU/GPU conflict (regionManager.h:122-131). (5) The download ring slot and dedicated staging stay alive until the callback runs (ring wraps wait priority ops, streamBuffer.cpp:326-328). Lock/ordering: the RangeSet and m_page_table/m_buffers/m_slot_buffers are GPU-thread-only (bufferCache.h:65). Tracker bits are only mutated under RegionManager::lock (a pure spin with no backoff, regionManager.h:36-41). Protection changes nest RegionManager::lock -> PageManager Region SpinGuard (pageManager.cpp:203) -> GuestAddressSpace::m_mutex (memoryAddressSpace.inc:748-755). WriteBacking takes only GuestBackingStore::m_mutex (memoryAddressSpace.inc:479).
- **How hot:** Profile-confirmed at >=38% of Thread_Gpu. One drain per GPU-write -> CPU-access cycle per 512 KiB window. GPU-dirty is set at RECORD time for the whole bound range of every written storage buffer (descriptors.cpp:142-143, bufferCache.cpp:476-479), for GPU-path DMA copy/fill destinations (bufferCache.cpp:532,564), and for anything a shader wrote. EOP labels are written eagerly at record time (graphicsRun.cpp:1194). So a game that waits on a label and then reads compute or DMA results (counters, culling or visibility output, DMA'd CPU staging) sees the label at once, touches the page and forces a wait for all work recorded so far. Every such read or write also costs one submit, a GPU drain, priority-thread wake-ups, WriteBacking memcpy and mprotect calls. Concurrent faulters on the same page each take a separate GPU-thread round trip. Their second trip finds no bytes and returns immediately, after the first has published.

- Idea (large): RECOMMENDED: asynchronous fault readback with per-page arm tokens; finalize runs in the priority callback.

(1) Tracker state. Add to RegionManager a per-page arm token: a lazily allocated uint64 array with 1024 entries (8 KiB per 4 MiB region), or a RegionBits mask plus token array. It is always read and written under RegionManager::lock. Add MemoryTracker::ArmReadback(runs, token), FinalizeReadback(runs, token) and ReadbackToken(page). Tokens come from a per-op monotonically increasing download sequence number, NOT from the tick. A small mutex-protected registry maps token -> {tick, runs}, like TextureCache::m_pending_downloads (textureCache.cpp:1949-1952, 1968-1976); it is used for waiter lookup, HasPendingReadback(range) and diagnostics.

(2) Re-dirty cancels. RegionManager::ChangeState<Gpu,true> clears the token of every page it marks, in the same critical section that sets m_gpu_dirty. It is reached only from ForEachUploadRange(is_written) (memoryTracker.h:120-127) and the unused MarkRegionAsGpuModified.

(3) DownloadBufferMemory (bufferCache.cpp:114-189) changes: iterate only GPU-dirty pages without a token; collect runs; Map the ring (this may wrap-wait and even Submit, so the tick can change); record the copy; DeferPriorityOperation; only then arm the collected pages with the new token and the final CurrentTick. Arming last guarantees the op is queued and the tick is final before any waiter can see the token.

(4) The priority callback does WriteBacking as today, then FinalizeReadback(runs, token). Under each region lock it clears m_gpu_dirty and the token only where the token still equals its own, then UpdateProtection (NoAccess -> Read). It never touches m_gpu_modified_ranges.

(5) Fault path, non-GPU threads only; GPU-thread callers and GPU-thread self-faults keep the synchronous path.
  a. Under the region lock, read {gpu_dirty, token} for the faulting page(s). If not GPU-dirty, return (retry the instruction).
  b. If a token is present, look up its tick without any GPU-thread round trip.
  c. Otherwise SendCommandSync a closure that: returns 0 if !IsRegionRegistered; runs FindBuffer and the window; calls DownloadBufferMemory; if it recorded, calls m_scheduler.Flush() (Submit+BeginNext, no wait; same command-buffer split the current Wait(CurrentTick) already makes at this point); and returns the tick armed on the faulting page.
  d. The faulting thread then calls MasterSemaphore::Wait(tick). Host wait-before-signal on a timeline value is legal, and Refresh/Wait use atomics (masterSemaphore.cpp:26-55). Do NOT call CommandScheduler::Wait, which calls CheckActive/Submit/BeginNext (GPU-thread-only). Then call WaitPriorityOperations(tick), which is legal off the priority thread (commandScheduler.cpp:305). Then return, and the instruction retries.
  e. Write faults and explicit InvalidateMemory loop on InvalidateRegion until every page is marked CPU-dirty under the lock, so kernel I/O gets RW pages.

(6) The synchronous path (GPU thread) computes wait_tick = max(recorded ? CurrentTick : 0, max armed tick in the window). It must wait even when nothing new was recorded; otherwise the MarkRegionAsCpuModified at :273 EXITs on still-GPU-dirty armed pages (UnmapMemory -> InvalidateMemory is the concrete case). If wait_tick < CurrentTick it is a partial wait, not a drain.

(7) Change HasGpuDirtyBytes (bufferCache.cpp:587-589) to return RangeSet.Intersects || tracker.HasArmed(range). This covers TryReadGpuCleanBacking (memory.cpp:881-889, shader-code reads via pipelineCache.cpp:97-100), TextureCache::SafeToDownload (textureCache.cpp:255-261; the image download prefills staging from backing at :1930 and later WriteBackings the whole range) and the image-insert MarkBufferModified (textureCache.cpp:1372-1375).

(8) GC (bufferCache.cpp:615-621) treats buffers that are dirty only through armed pages as dirty without EXIT_IF(!DownloadBufferMemory). Its Wait(CurrentTick) covers all armed ticks. Debug validators ValidateGpuDirtyPages/ValidateGpuDirtyOwnership (memoryTracker.cpp:19-47) must exclude armed pages.

Net effect: Thread_Gpu pays find + copy record + one vkQueueSubmit instead of a GPU drain, and keeps recording while the GPU and the priority thread publish.
  - Risks: Races and how the design prevents each:

(a) Other threads fault the same page while publication is pending. The page stays GPU-dirty and NoAccess until the callback finalizes it. Other faulters see the token under the region lock and wait on its tick with no GPU round trip. Without tokens they would busy-loop re-faulting: the RangeSet bytes are already subtracted, so DownloadBufferMemory returns false and nothing waits.

(b) The GPU thread records a new write to the page before finalize. ChangeState<Gpu,true> clears the token under the same lock, so finalize skips the page (no lost re-dirty). The guest re-faults and a new download of the new bytes (re-added to the RangeSet at bufferCache.cpp:478) is armed with a later token. The FIFO priority queue (pushed in nondecreasing tick order, commandScheduler.cpp:252, single runner) publishes old bytes before new ones.

(c) Two downloads arm the same page within one tick (download, re-dirty and download inside one command buffer). With tick-based arming the first callback would unprotect the page before the second WriteBacking and expose stale bytes. Unique per-op tokens prevent this.

(d) A waiter sees the arm before the op is queued or the tick is final (ring wrap inside Map can Submit). Arming last and waiting on the master semaphore first prevent this.

(e) CPU store racing WriteBacking. The page stays NoAccess until after WriteBacking inside the same callback. No other path unprotects GPU-dirty pages: InvalidateRegion checks GPU-dirty under the lock, UntrackMemory EXITs on GPU-dirty (memoryTracker.cpp:219-223), and texture untracking only removes write watchers. A guest mprotect (KernelMprotect, memory.cpp:~3805) can still override tracking protections; this is pre-existing.

(f) Stale backing read on Thread_Gpu. Every consumer keyed on byte-dirty must see armed pages (point 7). Tracker-based checks (ObtainBuffer small path :459-461, ObtainBufferForImage :496, CopyBuffer/FillBuffer fast paths :524,549) already see armed pages as GPU-dirty.

(g) Deadlock and lock order. The callback takes GuestBackingStore::m_mutex (released), then RegionManager::lock -> PageManager spin -> GuestAddressSpace::m_mutex, the same order as all existing users. There is no inversion as long as no thread waits for priority ops while holding a RegionManager lock. That holds today: the is_written upload holds region locks across UploadCopies, which only Maps the Upload ring, and Upload wraps never WaitPriorityOperations (streamBuffer.cpp:326). Add an assert (WaitPriorityOperations EXIT if the calling thread owns a region lock or s_upload_owner != nullptr). A residual cost remains: the priority thread may spin (TrackingSpinLock has no yield) while the GPU thread holds region locks through an upload that itself waits on the GPU for a staging wrap. That delays EOP interrupts and flip completion queued behind it; mitigate with a try-lock plus re-queue, or by pre-mapping staging before taking the locks.

(h) Unmap while pending. UnmapMemory's Finish+WaitPriorityOperations (renderContext.cpp:115-119) runs all callbacks before InvalidateMemory, and the kernel unmaps only after that. A late waiter finds no token and simply returns.

(i) Buffer merge or GC while pending. The callback captures only staging plus addresses; old source buffers are erased via DeferOperation at a later tick (bufferCache.cpp:102-112). GC must clear tokens when it unmarks.

(j) Shutdown. ShutdownGpu's Finish+DrainPriorityOperations (renderContext.cpp:40-43) finalizes every token, so the ~BufferCache GPU-dirty EXIT (bufferCache.cpp:226-235) stays valid. This is why the callback, not the waiter, should finalize.

(k) Livelock if the GPU keeps re-dirtying a polled page. This is inherent; today's code has the same retry semantics, because HandleFault returning true only retries the instruction.

(l) Tests call ReadMemory from the test thread and read backing right after (tests/ShaderRecompilerComputeTests.cpp:3939-3958). The caller must still block until finalize, which this design does.

(m) Extra vkQueueSubmit per readback. Coalesce (see the next idea).

(n) Benefit is bounded when the faulting guest thread is itself on the frame's critical path. It still waits for everything recorded before the copy (single queue, in-order), but Thread_Gpu keeps working.
- Idea (medium): Variant literally as posed: the waiting (faulting) thread performs the finalize itself after MasterSemaphore::Wait + WaitPriorityOperations. It calls FinalizeReadback(runs, token) under the region lock, then, for writes, marks CPU-dirty only if the page is no longer GPU-dirty, in the same critical section.
  - Risks: It needs the same per-page tokens and the same re-dirty cancel; otherwise UnmarkRegionAsGpuModified(window) run on another thread loses a re-dirty recorded after the download. Unlike the callback version, arms linger if the waiter is a thread that never returns (suspended, killed, shutdown). Lingering arms trip the GC EXIT (bufferCache.cpp:620, 639-642), the destructor EXIT (226-235) and the MarkRegionAsCpuModified conflict EXIT on Unmap. So GC, Unmap and Shutdown must also finalize, which splits finalize across four sites. It also cannot finalize pages armed by someone else's op unless it waited for that op's tick, and a same-tick double arm needs tokens anyway. The callback variant is strictly simpler.
- Idea (small): Coalesce submits. Faulting-thread closures processed in one ProcessCommands batch (graphicsRun.cpp:129-142) record their downloads and set a flush-requested flag. ProcessCommands issues one Flush() after the loop, then releases all the callers' semaphores; the closure returns before done.release() and a batch completion releases them.
  - Risks: The caller must not be released before the tick is submitted, or MasterSemaphore::Wait blocks until the next natural submit, which may be a whole guest submission away. Changing the SendCommandSync release point touches the generic command lane that UnmapMemory and TextureCache::InvalidateMemory also use.
- Idea (medium): Proactive (speculative) download of CPU-consumed GPU writes, built on the arm/finalize machinery. Remember pages that recently read-faulted (a per-page hint bit set in HandleFault) and DMA destinations in guest-visible CPU memory (CopyBuffer/FillBuffer GPU path, bufferCache.cpp:531-533, 556-566). When such a range is re-dirtied, record its download right after the producing command, at the next Flush or EOP. The page is then published and finalized by the priority callback. The EOP interrupt callback is queued later in the same FIFO (sync.cpp:231-256), so a game that waits on the interrupt finds the page already clean: no fault, no wait. A game that polls the eager label faults, but finds a token and waits only for that earlier tick.
  - Risks: Wasted PCIe bandwidth, download-ring pressure and priority-thread memcpy for data the CPU never reads. It needs a budget and hysteresis (drop the hint after N unused downloads). Re-dirty churn: every re-dirty cancels the arm, so this only helps if the GPU write and the CPU read alternate. Ring pressure increases wrap waits (streamBuffer.cpp:325-328).
- Idea (small): Make the 512 KiB widening window adaptive. Download the whole GPU-dirty extent of the owning buffer when it is small (<= a few MiB). Grow the window for sequential faulters (last fault address + 512 KiB) so one wait covers a whole streaming read instead of one drain per 512 KiB.
  - Risks: Larger copies lengthen the single wait and the priority-thread memcpy. Pages made clean that the CPU never touches will be re-uploaded as CPU-dirty only if the CPU writes them; clean pages cost nothing on the GPU side. There is no correctness risk: the window is clamped to the buffer (bufferCache.cpp:263-264) and only dirty bytes are copied.

### Wait site: GPU-thread consumers that need the bytes immediately (synchronous ReadMemory on Thread_Gpu): textureCache.cpp:1211-1214 TextureCache::MaterializeDccClear; graphicsRun.cpp:823-846 CommandProcessor::SynchronizePredicate; GPU-thread self-faults at graphicsRun.cpp:1019/1029/1120 (indirect args), :1194 (EOP label), :388 (reference clock)

- **Kind:** Same full drain as bufferCache.cpp:266-271, executed inline on Thread_Gpu (SendCommandSync short-circuits, graphicsRun.cpp:146-149). SynchronizePredicate additionally drains conservatively with BufferFlushAndWait + WaitPriorityOperations(CurrentTick-1) when an image producer or pending image writeback exists (graphicsRun.cpp:836-841).
- **Thread:** Thread_Gpu only.
- **Reached from:** PM4 draw packet -> texture binding -> TextureCache::FindImage (textureCache.cpp:1305..1387) -> MaterializeDccClear (1179) -> BufferCache::IsRegionGpuModified(dcc) -> ReadMemory (1213) -> TryReadBacking of the DCC code (1219-1228) -> optional ClearImage + FillBuffer(UINT32_MAX) (1242); PM4 SET_PREDICATION op 0x03 wait_op!=0 (graphicsRun.cpp:876-881) -> SynchronizePredicate -> ReadMemory (844) -> CPU read of the 64-bit predicate (881); PM4 DRAW_INDIRECT / DRAW_INDEX_INDIRECT(_MULTI) -> CPU deref of args at graphicsRun.cpp:1019/1029. If a compute shader wrote the args (GPU-driven culling), the page is NoAccess -> fault on Thread_Gpu -> HandleFault -> ReadMemory inline; PM4 EOP with eager label memcpy (graphicsRun.cpp:1194) or WriteReferenceClock (388) into a GPU-dirty page -> write fault on Thread_Gpu -> InvalidateRegion -> ReadMemory(is_write) inline
- **Protects:** The CPU-side emulation reads exact current bytes: the DCC clear code, the predicate value, and the indirect draw counts and firstIndex used to compute the host index pointer (graphicsRun.cpp:1031-1036). For label writes, the CPU store must not be overwritten by a later publication, and the page must end CPU-dirty.
- **How hot:** Unknown split; this is the likely source of the texture-lookup trigger (2). MaterializeDccClear runs on every FindImage of a DCC surface and drains whenever the metadata range is GPU-dirty, for example metadata written by a compute or GPU-path fill; that can happen per surface per frame. Indirect-args faults would drain once per indirect packet if Astro Bot uses GPU-generated draw args. The asynchronous design above does NOT help these, because Thread_Gpu itself needs the data.

- Idea (large): GPU-side indirect draws. Instead of dereferencing args on the CPU, bind ObtainBuffer(args_addr, stride*count, false), as DispatchIndirect already does (renderCompute.cpp:435-436), and use vkCmdDrawIndirect/vkCmdDrawIndexedIndirect with the whole guest index buffer bound at m_index_base_addr, so firstIndex comes from the args.
  - Risks: CPU-side behavior would be lost. It currently clamps index_count to m_index_buffer_size (graphicsRun.cpp:1033-1036); the GPU needs robustBufferAccess2 or a compute clamp pre-pass. It also sets m_num_instances from args (1020), and it computes index_addr per draw for host index handling (1031), including index-format conversion and primitive-restart emulation, if any. Mesh-dispatch slicing and other CPU-derived per-draw state would also need GPU equivalents.
- Idea (medium): DCC clear materialization without readback. Remember the last value written to a metadata range when it is known at record time: the FillBuffer GPU path (bufferCache.cpp:531-533) and MaterializeDccClear's own FillBuffer(UINT32_MAX). Invalidate that knowledge on any other GPU write overlapping the range: ObtainBuffer(is_written) overlap, CopyBuffer dst, image aliases via InvalidateMemoryFromGPU. Invalidate it on CPU writes (InvalidateMemory). MaterializeDccClear uses the known value and calls ReadMemory only when it is unknown.
  - Risks: A missed invalidation path produces a wrong clear color or a skipped clear. Shader storage writes via BDA are not tracked as GPU-dirty at all today (only ObtainBuffer(is_written) marks), so metadata written that way would also be missed by the current code.
- Idea (medium): Predicates via VK_EXT_conditional_rendering: keep the predicate on the GPU and skip draws there, instead of reading it back.
  - Risks: Semantic mismatch. The guest predicate is 64-bit with condition 0/1, while Vulkan tests a 32-bit value for non-zero (inversion is supported). m_predicate_skip also affects CPU-side packet processing, which cannot be skipped on the GPU. The branch's producer-aware classification (graphicsRun.cpp:833-845) already avoids most drains.
- Idea (small): After the async design lands: when every GPU-dirty page in the window is already armed by a flushed download, the synchronous path waits only for that older tick. CommandScheduler::Wait with tick < CurrentTick does not submit or drain (commandScheduler.cpp:205-207). This removes the full drain for GPU-thread consumers that follow a guest fault on the same data.
  - Risks: You must still WaitPriorityOperations(that tick) and re-check for re-dirty; a new write recorded since then means the new bytes must be downloaded at CurrentTick.

### Wait site: src/graphics/host_gpu/renderer/cache/bufferCache.cpp:632-646 BufferCache::RunGarbageCollector (dirty eviction)

- **Kind:** Full drain: Wait(CurrentTick()) (submit open command buffer + wait) + WaitPriorityOperations(CurrentTick) after DownloadBufferMemory for up to 64 dirty victims (:619-621). Then Unmark, verify, UntrackMemory, Unregister and immediate erase (:636-646).
- **Thread:** Thread_Gpu. Called from RenderContext::RunGarbageCollector (renderContext.cpp:147-155), which runs at completed Graphics/Compute submissions (graphicsRun.cpp:610,614,636,640) and at FlipPreparation (645).
- **Reached from:** GuestGpu::ThreadRun -> GuestGpu::Process -> (complete submission) m_renderer.RunGarbageCollector() -> ProcessFaultBuffer / TextureCache::ProcessDownloadImages / TextureCache::RunGarbageCollector / BufferCache::RunGarbageCollector
- **Protects:** Victim buffers must be published to the backing, their pages must stop being GPU-dirty (checked at :639-642), and they must be CPU-dirty (untracked, :643) before Unregister/erase (:644-645). Otherwise the only copy of GPU-written bytes is destroyed, or a CPU access would find no owner for a GPU-dirty page. The immediate erase is also only safe because the GPU is idle after the drain.
- **How hot:** Rare. Only when device memory usage (VMA budget, :597-599, which includes images) is >= m_critical_gc_memory. That threshold is about budget minus 1.6 GiB (:214-222); below it, dirty buffers are skipped (:616-618). Runs at most once per completed submission.

- Idea (medium): Asynchronous dirty eviction. Record downloads (pages armed and finalized by the callback per the async design) and move victims to a retiring list with {tick, token}. On later GC passes, retire the entries whose tick IsFree and whose priority ops are complete. Retirement needs a new non-blocking CommandScheduler::PriorityOperationsDone(tick), which is the WaitPriorityOperations predicate evaluated once under m_operation_mutex. Before retiring, re-check that the buffer is not GPU-dirty and has no RangeSet bytes; if it was re-dirtied or touched (LRU), drop it from the list.
  - Risks: Memory is released a few submissions later while already at critical pressure, which could lead to allocation failure (VMA WITHIN_BUDGET, streamBuffer.cpp:73-74, EXIT on failure at :88). A buffer reused by the GPU thread between record and retirement must be cancelled, not freed, or GPU-written state is lost. Retirement order must stay publish -> finalize -> UntrackMemory -> Unregister; Unregister first would let CPU stores race WriteBacking. Deferred erase is needed because the GPU may still use the buffer.

### Wait site: src/graphics/host_gpu/renderer/cache/faultManager.cpp:78-81 FaultManager::ProcessFaultBuffer

- **Kind:** Waits for a specific older tick: m_scheduler.Wait(m_fault_areas[m_current_area]), the tick of the area used 8 calls earlier (ring size MaxPendingFaults=8, faultManager.h:15). If that tick is still the open command buffer it becomes Submit + full drain (commandScheduler.cpp:196-204). It is followed by PopPendingOperations (commandScheduler.cpp:210-226), which runs every completed normal deferred op and calls WaitPriorityOperations(op.tick) before each.
- **Thread:** Thread_Gpu.
- **Reached from:** PrepareGraphicsBindings / compute dispatch with program.info.uses_dma (descriptors.cpp:899-901, renderCompute.cpp:372-373,430-431) -> RenderContext::PrepareBda sets m_fault_process_pending (renderContext.cpp:144) -> submission complete -> RenderContext::RunGarbageCollector (renderContext.cpp:148-151) -> BufferCache::ProcessFaultBuffer (bufferCache.cpp:649-651) -> FaultManager::ProcessFaultBuffer
- **Protects:** The 8 KiB host-visible download area (:83-86) is memset and overwritten by a new fault-parse dispatch. It must not be reused until the GPU has finished writing it and the deferred parser callback (:131-145, which also calls FindBuffer for faulted pages and zeroes m_fault_areas[area]) has consumed it.
- **How hot:** Once per completed guest submission that used BDA shaders. It blocks only when the GPU is >= 8 such submissions behind, so it caps how far Thread_Gpu can run ahead in submission count. That cap is cheap at 40% GPU load unless the game issues many small submissions per frame. The PopPendingOperations part can also block on the priority thread (for example on large image WriteBacking) via the per-op WaitPriorityOperations.

- Idea (small): Do not block. If the area's tick is not IsFree (a non-blocking Refresh), either grow the ring (areas are only 8 KiB; allocate more, the way CommandPool::Grow does, commandScheduler.cpp:48-60) or skip this round and keep m_fault_process_pending set. Skipping is safe for the device-side state: the page-fault bitmask persists, and the shader clears only the words it scans (src/graphics/host_gpu/shaders/fault_buffer_process.comp).
  - Risks: Skipping delays creating buffers for pages first touched via BDA; those pages read through a null page-table entry, which is already a one-submission delay. Growing costs memory. Pre-existing bug: the shader clears a word before checking MAX_PAGE_FAULTS, so faults beyond 1023 per round are dropped permanently until re-accessed. Running the parser less often makes each round more likely to overflow.

### Wait site: src/graphics/host_gpu/renderer/cache/streamBuffer.cpp:320-331 StreamBuffer::WaitPendingOperations (Wait at :325; WaitPriorityOperations at :326-328 for Download usage)

- **Kind:** Waits for specific older ticks: for each watch of the previous lap covering the requested bytes, Scheduler().Wait(watch.tick). If watch.tick == CurrentTick (the ring lapped inside one command buffer) it is Submit + full drain. The Download ring also does WaitPriorityOperations(watch.tick). With allow_wait=false it returns nullptr instead (the ObtainBuffer small path, bufferCache.cpp:464).
- **Thread:** Thread_Gpu (every Map caller). Possibly while holding RegionManager locks: UploadCopies -> m_staging_buffer.Map inside ForEachUploadRange(is_written) (memoryTracker.h:111-127, bufferCache.cpp:427), and while holding TextureCache::m_lock (DownloadImageMemory, textureCache.cpp:1914-1915).
- **Reached from:** BufferCache::DownloadBufferMemory -> m_download_buffer.Map (bufferCache.cpp:133); BufferCache::UploadCopies -> m_staging_buffer.Map (bufferCache.cpp:427), reached from SynchronizeBuffer/ObtainBuffer; BufferCache::WriteDataBuffer -> m_staging_buffer.Copy (bufferCache.cpp:38); TextureCache::DownloadImageMemory -> Download ring Map (textureCache.cpp:1914-1915)
- **Protects:** Ring bytes are not overwritten while a submitted copy still reads (Upload/Stream) or writes (Download) them. For Download, the priority callback must have finished reading the mapped bytes into the backing (bufferCache.cpp:181-185).
- **How hot:** Low to moderate. Rings: staging 512 MiB, stream 64 MiB, download 64 MiB (bufferCache.cpp:198-201). There is one watch per tick (Commit, :291-301), so wrap waits usually target ticks a frame or more old that are already complete. It becomes a full drain only if a ring laps inside one command buffer. The async readback design adds a Flush per readback, which makes ticks finer and lowers this risk.

- Idea (small): When the needed watch is not free, fall back to a dedicated temporary Buffer instead of waiting (as DownloadBufferMemory does for oversize requests, bufferCache.cpp:136-145, and UploadCopies at 438-448). Always do this when watch.tick == CurrentTick. Alternatively, grow the ring by chaining a second block.
  - Risks: Temporaries must outlive both the GPU copy and, for Download, the priority callback: they are owned by the callback or released via DeferOperation. The extra allocations can fail under VRAM or host budget pressure. Map is called under RegionManager locks in the upload path, so an allocation there lengthens the time other threads spin on those locks.

### Facts

- Only one path sets GPU-dirty: ObtainBuffer(is_written) -> SynchronizeBuffer -> MemoryTracker::ForEachUploadRange(is_written=true) (bufferCache.cpp:476, 382-388; memoryTracker.h:103-129). MarkRegionAsGpuModified (memoryTracker.cpp:189-195) has no callers. For is_written, the region locks are held from the CPU-dirty scan (:112) through UploadCopies (:119) to the GPU mark (:120-127). The bytes are added to m_gpu_modified_ranges afterwards (bufferCache.cpp:477-479). All of this runs on Thread_Gpu at record time, before the GPU executes the write.
- Protection model (regionManager.h:143-147, 171-181; pageManager.cpp:84-96):
- CPU-dirty page: no watchers, ReadWrite.
- Clean page: buffer write watcher, Read-only.
- GPU-dirty page: write + access watcher, NoAccess.
PageManager counts write watchers (7 bits) shared with TextureCache's TrackImage/UntrackImage (textureCache.cpp:397-502). The access watcher is 1 bit and used only by buffer GPU-dirty state (overflow is fatal, pageManager.cpp:102-105). New RegionManagers start fully CPU-dirty, readable and writable (regionManager.h:98-100).
- Who changes protection:
- Protect: Thread_Gpu at record time. The upload clears CPU-dirty and adds write watchers (regionManager.h:155-161); the GPU mark adds access watchers.
- Unprotect: faulting threads in InvalidateRegion when the page is not GPU-dirty (memoryTracker.h:67-72); Thread_Gpu in ReadMemory (Unmark :270, MarkCpu :273); GC (:638, :643); UntrackMemory (memoryTracker.cpp:205-227).
All changes happen under RegionManager::lock and nest PageManager::Region SpinGuard (pageManager.cpp:203) -> ProtectGuestHostMemory -> GuestAddressSpace::m_mutex (memory.cpp:3749-3752, memoryAddressSpace.inc:748-755).
- RegionManager::lock is TrackingSpinLock. It is a pure test_and_set spin with no yield or backoff, and it EXITs on recursive acquisition by the same thread id (regionManager.h:29-50). Multi-region lockers (ForEachUploadRange, UntrackMemory) acquire locks in ascending address order.
- Publication path: DeferPriorityOperation (commandScheduler.cpp:247-266) queues {callback, CurrentTick()} FIFO. A single priority jthread (268-295) pops the front, does MasterSemaphore::Wait(tick) (possibly on a not-yet-submitted open command buffer tick), then runs the callback. The callback calls Buffer::Invalidate (vmaInvalidateAllocation, only if non-coherent, streamBuffer.cpp:131-138) and then LibKernel::Memory::WriteBacking (memory.cpp:909-915) -> GuestAddressSpace::TryWriteBacking -> GuestBackingStore::TryTransferBacking. That memcpys into the m_backing_base alias under GuestBackingStore::m_mutex (memoryAddressSpace.inc:474-500, 726-728). The alias write never faults, whatever the guest-view protection.
- WaitPriorityOperations(tick) (commandScheduler.cpp:304-313) blocks on m_operation_mutex/condvar until no op with tick <= T is active or at the queue front. It is callable from any thread except the priority thread or a deferred callback (EXIT at :305). MasterSemaphore::Wait/Refresh/IsFree are thread-safe (atomics + vkGetSemaphoreCounterValue/vkWaitSemaphores, masterSemaphore.cpp:26-55). CommandScheduler::Wait/Flush/Submit/Current/DeferOperation are GPU-thread-only (they call CheckActive, mutate m_command, and push with the open-command-buffer tick).
- CommandScheduler::Wait(tick) is a full drain only when tick == CurrentTick() (Submit + wait + BeginNext, commandScheduler.cpp:196-204). Waiting on an older tick is a partial wait (:205-207). Finish waits for CurrentTick-1 and runs PopPendingOperations (184-192).
- ReadMemory's 512 KiB window: window_begin = max(AlignDown(vaddr, 512K), buffer_begin); window_end = min(max(window_begin+512K, vaddr+size), buffer_end) (bufferCache.cpp:259-264). Only GPU-dirty pages are scanned, and only their dirty bytes are copied (64-byte-aligned packing, :122-126). Write faults mark only [vaddr,size] CPU-dirty; the rest of the window becomes clean (Read-only). Tests pin the clamped half-open window semantics (tests/ShaderRecompilerComputeTests.cpp:3920-3958).
- Invariant used by debug validators (memoryTracker.cpp:19-47): a tracker page is GPU-dirty iff it has bytes in m_gpu_modified_ranges. Today it is violated only transiently between DownloadBufferMemory's Subtract (bufferCache.cpp:127) and Unmark (:270, :638), with no GPU-thread recording in between. Any async design makes 'GPU-dirty with no bytes' (armed) a persistent state, so these validators and the GC EXIT_IF(!DownloadBufferMemory) at :620 must be updated.
- HasGpuDirtyBytes (bufferCache.cpp:587-589) consults only the byte RangeSet. Three GPU-thread consumers use it as 'the backing is current', and all would read stale backing during an async pending window:
- TryReadGpuCleanBacking (memory.cpp:881-889), used by ReadShaderGuestMemory (pipelineCache.cpp:97-100).
- TextureCache::SafeToDownload (textureCache.cpp:255-261). DownloadImageMemory prefills staging from backing at record time (:1930) and later WriteBackings the whole range (:1953-1965).
- FindImage insert (textureCache.cpp:1372-1375).
Tracker-based checks (IsRegionGpuModified) remain correct.
- Memory types (streamBuffer.cpp:20-41, 58-105): DeviceLocal (all cached game buffers, BDA page table, fault bitmask) uses VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, flags WITHIN_BUDGET, plus DEDICATED when eShaderDeviceAddress (every cached game buffer, bufferCache.cpp:362-364), with no MAPPED or HOST_ACCESS bits and empty preferredFlags. VMA's no-CPU-access branch prefers DEVICE_LOCAL and penalizes only DEVICE_UNCACHED_AMD (vk_mem_alloc.h:4331-4362). The cost function picks the first zero-cost type (14306-14318), normally the plain VRAM type ahead of the BAR type. Even if a DEVICE_LOCAL|HOST_VISIBLE type were chosen, the buffer is never mapped (m_mapped stays empty, streamBuffer.cpp:101-104). Stream is AUTO_PREFER_DEVICE with SEQUENTIAL_WRITE and MAPPED, so it may land in BAR. Upload is AUTO_PREFER_HOST sequential. Download is AUTO_PREFER_HOST with RANDOM access and MAPPED, preferring HOST_CACHED (and HOST_COHERENT via preferredFlags). ReBAR direct readback would not remove the wait: the bytes still must reach the guest backing, and CPU reads of WC VRAM are slow.
- The fault handler runs on the faulting thread: KytyExceptionHandler (runtimeLinker.cpp:782-802) -> HandleGpuFault (memory.cpp:929) -> RenderContext::HandleFault (renderContext.cpp:58-72, 1-byte size, IsMapped gate under a shared_mutex). Returning true simply re-executes the instruction, so any design can rely on retry: re-faulting on a still-protected page is correct.
- ReadMemory refuses to run from a non-GPU thread inside a deferred callback (bufferCache.cpp:248-252). For read faults it does not check IsRegionRegistered, so FindBuffer may CREATE a buffer (merging overlaps) for a page that GC just released (bufferCache.cpp:257, 278-293). For writes it returns early if the region is unregistered (254-256).
- EOP label writes are eager: CommandProcessor::WriteAtEndOfPipe memcpys the value into guest memory at PM4 processing time (graphicsRun.cpp:1194), while the interrupt is a priority callback at GPU completion (sync.cpp:245-256). A game polling labels therefore sees completion before the GPU work runs. Its subsequent reads of GPU-written buffers are exactly the fault-driven readbacks.
- Faulting-thread commands are serviced only between PM4 packets (graphicsRun.cpp:697-700) or when Thread_Gpu is idle (531-541). While Thread_Gpu is blocked in any wait (a drain, FaultManager, WaitFlipDone), SendCommandSync callers queue behind it. Pre-existing deadlock hazard: a thread Thread_Gpu waits on (for example the VideoOut path) that faults on a GPU-dirty page.
- TextureCache::InvalidateMemory (write faults, textureCache.cpp:1713-1744) uses the same GPU-thread round trip plus Wait(CurrentTick) + WaitPriorityOperations (1732-1738) whenever an image writeback is pending on the page. The off-GPU-thread wait pattern applies there too: the GPU thread only Flushes if the pending tick is still open, and the faulting thread waits itself. The texture cache already has the registry (m_pending_downloads, 1949-1952, 1968-1976) but stores no tick.
- RenderContext::UnmapMemory (renderContext.cpp:97-132) does Finish + WaitPriorityOperations before BufferCache::InvalidateMemory, so every pending publication completes before unmapping. ShutdownGpu does Finish + DrainPriorityOperations (40-43). The CommandScheduler is destroyed after BufferCache and TextureCache only after an explicit Shutdown joins the priority thread (renderContext.cpp:23-26), so callbacks that capture 'this' are safe.
- Kernel I/O pre-invalidation (fileSystem.cpp:162,622,...) needs the whole range CPU-dirty (RW) on return, because host ReadFile/recv into a protected page do not fault. An async design must loop until CPU-dirty is marked, not just retry the instruction. Even today, a page can be re-protected by Thread_Gpu between InvalidateMemory returning and the host write.
- BufferCache::m_gpu_modified_ranges, m_page_table, m_buffers, m_slot_buffers and m_lru_cache are unsynchronized and GPU-thread-only (bufferCache.h:65, 125-129). RangeSet is a plain std::map (rangeSet.h:12-60).
- The fault parser shader (src/graphics/host_gpu/shaders/fault_buffer_process.comp) zeroes each bitmask word before checking MAX_PAGE_FAULTS. Faults beyond 1023 per round are silently lost.

### Open questions

- What is the real split of ReadMemory calls and wait time by caller: guest read fault, guest write fault, kernel InvalidateMemory, MaterializeDccClear, SynchronizePredicate, GC, and Thread_Gpu self-faults (indirect args at graphicsRun.cpp:1019/1029/1120, EOP label at 1194)? Only the guest-thread share benefits from moving the wait off Thread_Gpu. Instrument with per-caller counters, bytes, and wall time spent in Wait/WaitPriorityOperations.
- Which guest threads fault, and are they on the game's frame-critical path? If the faulting thread is the game's render-submit thread, async readback frees Thread_Gpu but the game thread still waits for everything recorded before the copy. The frame-rate gain then depends on how much already-submitted PM4 Thread_Gpu can process meanwhile.
- Does Astro Bot poll eager EOP labels or wait on EOP interrupts before reading GPU-written buffers? This decides whether proactive download can eliminate faults, or only turn them into short token waits.
- Does Astro Bot use GPU-generated indirect draw args or DCC metadata written by compute? Either would put synchronous drains on Thread_Gpu that the fault-path redesign cannot remove.
- Which Vulkan memory type index does VMA pick for DeviceLocal buffers on the user's RX 9070 XT Windows driver (8389003)? Log vmaGetAllocationMemoryProperties or LogMemoryBudget. It is expected to be non-host-visible, but this was not verified on the device.
- How often does re-dirty happen between download and finalize (pages armed then cancelled)? That measures the livelock/retry risk and how much proactive downloading would waste.
- How many vkQueueSubmits per frame would a Flush per async readback add, and what does one submit cost on the AMD Windows driver? This decides whether coalescing is required.
- Are shader writes through BDA (uses_dma programs) tracked as GPU-dirty at all? Nothing calls MarkRegionAsGpuModified, and PrepareBda only uploads CPU-dirty data. If such writes exist, CPU reads of that memory see stale backing regardless of this design.
- Is Config::ReadbackLinearImagesEnabled on in the user's run? If so, TextureCache::ProcessDownloadImages queues image writebacks every GC round, which makes the TextureCache::InvalidateMemory pending-download drain more frequent.
- Is TrackingSpinLock contention measurable when the priority thread finalizes while Thread_Gpu holds region locks across UploadCopies, which can wait on a staging-ring wrap?

## Shader translation and pipeline creation (Performance survey)

Shader and pipeline creation on this branch is almost entirely synchronous on Thread_Gpu. Each draw calls RefreshShaders (renderDraw.cpp:867-895) and then PipelineCache::GetGraphicsPrograms (pipelineCache.cpp:687-763). That function prepares the stage inputs, takes m_mutex, and runs ProgramCache::Get for the PS first and then for the VS/LS/HS/TES stages. Get builds a static-input key vector (shader.cpp:653-759) and finds the source entry. It then re-runs MaterializeResources on every draw, which is a two-walker SRT evaluation reading guest memory, and does a linear search of the permutations for a matching ResourceSpecialization plus push-data start (pipelineCache.cpp:282-315).

On a miss, the whole frontend runs again, even when the source is already cached (TranslateProgram at :356): decode, CFG build and structurize, IR, SSA, constant propagation, DCE, ReadLane, SRT plan and resource tracking (ShaderRecompiler.cpp:486-628). After that come CompileProgram (specialize, collect info, allocate bindings, emit SPIR-V; :630-668), optional spirv-val (:150-178), vkCreateShaderModule, a journal append with a flush, and an unconditional printf that recounts every program (:379-391).

GetGraphicsPipeline (:780-951) builds a GraphicsPipelineKey of about 720 bytes and hashes the 129 bytes of static state one byte at a time (pipelineCache.h:198-233). On a miss, CreatePipelineInternal creates a per-pipeline descriptor set layout, a pipeline layout, and, for rect lists, freshly emitted TCS/TES modules. It then calls vkCreateGraphicsPipelines synchronously (shaders.cpp:209-567). Compute works the same way, keyed only by program id.

The two persistent caches are a VkPipelineCache and the shader journal. Both are disabled unless four conditions hold: CMAKE_BUILD_TYPE is not Debug, the git hash is known and not "-dirty", a TITLE_ID exists, and the journal is not switched off (pipelineCache.cpp:438-456, 524-529). Both are keyed on the full git revision plus the driver identity, and the journal also on APP_VER. The driver cache is written only on a clean window exit (window.cpp:820) and then destroyed (:683-684), so a crash discards the whole session.

The journal stores shader permutations only, never pipelines. Replay runs on one thread that holds m_mutex across translate and compile (:589-619). The first draw or dispatch waits for the entire replay (WaitForPrecompile, :544-549, 692, 768).

Several state values create permutations without need:
- A VS's push-data start depends on the size of the PS compiled before it, so depth-only, shadow and main passes of the same VS each become separate full compiles.
- Clip-disabled draws bake the viewport scale and offset into VS SPIR-V constants.
- Buffer stride, format and swizzle are baked into the specialization.
- Depth-bounds min/max stay in the pipeline key even when the test is off. Cull mode, front face, topology and primitive restart are static, although Vulkan 1.3 makes them dynamic.

The recompiler has no mutable global state, so it can run on several threads. Materialization cannot, because TryReadGpuCleanBacking returns false off the GPU thread (memory.cpp:883), and the plan's evaluation scratch belongs to the GPU thread (ShaderIR.h:562). Upstream PR #35 targets the old render-pass architecture and is unsound; only its idea of saving the cache periodically is worth keeping. No Tracy zones cover translate, emit, validate or vkCreate*, so the 12% figure cannot yet be broken down by phase.

### Proposal (small): Make the shader journal and driver pipeline cache valid for real builds, and survive crashes

**What.** (1) Replace the rule that disables caching for dirty builds (pipelineCache.cpp:453-456) with a build-identity key. For a dirty build, use an XXH3 of the running executable (or of a source hash computed at configure time) in place of KYTY_GIT_REVISION in DriverCacheSignature (:64-74). (2) Stop keying the journal on the full git revision. Instead key it on KYTY_SHADER_ABI_HASH, a SHA-256 that generate_version.cmake computes over src/graphics/shader/**, src/graphics/host_gpu/renderer/pipeline/shaderPrecompile.*, shaderCompiler.h, shader.h and ShaderIR.h, plus FormatVersion. (3) Drop the emulator revision from the driver-cache signature entirely. The driver content-addresses entries by SPIR-V and state, so keep vendor, device, driverVersion, pipelineCacheUUID and the XXH3 payload check, and reset the file when it grows past about 512 MB. (4) Split PipelineCache::Save into SnapshotDriverCache(), which does not destroy the cache, and Shutdown(). Run the snapshot on a low-priority background thread every 60 s when at least one pipeline was created since the last save, reusing the existing temp-file, flush and rename writer (:664-680). vkGetPipelineCacheData is safe to call while other threads create pipelines against the same internally synchronized cache.

**Why.** The doc states that a modified, uncommitted build disables both caches. The code enforces this at pipelineCache.cpp:443-456 and 527-529, and generate_version.cmake:30-38 appends -dirty for any uncommitted change. Every commit, even a docs-only one, invalidates both files because KYTY_GIT_REVISION is in the signature (:71). The driver cache is saved only at WindowRun exit (window.cpp:820), so any EXIT or crash, which is common in a hobby emulator, throws away every pipeline compiled that session. The journal survives crashes because it flushes each record (shaderPrecompile.cpp:424), but it is useless while the gate is closed. If the Astro Bot profile came from a dirty or freshly committed build, every run paid full translation cost.

**Impact.** On second and later runs of the same binary, this removes almost all recompiler time (TranslateProgram and CompileProgram) from Thread_Gpu. That is the bulk of the ~12% if the profile was taken with the gate closed, and it also removes the hitches in the 20-40 fps range that come from first-seen shaders. Keeping the driver cache across commits and crashes avoids cold vkCreateGraphicsPipelines calls, which typically take 10-100 ms each on a cold cache. AMD's own disk cache (VkCache) may already hide part of this on Windows. Expect roughly 0-12% more steady-state fps and a large improvement in 1% lows on repeat runs.

**Risks.** Replaying stale records after a recompiler change. Replay recompiles from guest code with the current compiler, so the modules themselves are correct. The danger is a new input field that affects compilation but is not serialized by Fields()/Specialization() (shaderPrecompile.cpp:125-215) or included in BuildStageStaticKey: a record would then compile with a default value, and a live draw could match it and bind a wrong module. Mitigations: put the headers that define those structs in the ABI hash, and add a static_assert on sizeof(ShaderVertexInputInfo/ShaderPixelInputInfo/ShaderComputeInputInfo) next to Fields() so a new member fails the build until FormatVersion is bumped. A driver cache that is never pruned grows over time, so cap its size. A periodic save racing with Save() at shutdown needs a mutex.

**Validation.** Extend shader_precompile_record_tests: a key built from the ABI hash must reject a mismatched journal. Add a struct-size guard test. Run Astro Bot twice on the same dirty build: the second run should log 'Shader precompile: replaying N' and show near-zero time in a new ShaderCompile Tracy zone. Kill the process with taskkill /F mid-session and check that _PipelineCache/<id>.bin exists, passes the XXH3 check, and makes vkCreateGraphicsPipelines faster on the next run (zone timing).

### Proposal (small): Instrument compile phases and log why each new permutation was created

**What.** Add KYTY_PROFILER_BLOCK zones around TranslateProgram, CompileProgram, ValidateShaderSpirv, CompileSPV, ProgramCache::Get's MaterializeResources, CreateDescriptorLayout, createPipelineLayout, createGraphicsPipelines/createComputePipelines, and Journal::Append. Split 'PipelineCache::CreatePipeline(Gfx)' (pipelineCache.cpp:789) into separate lookup and create zones; today the one zone covers the per-draw lookup as well. Add a diagnostic flag, --shader-permutation-log. When a key hits an existing SourceEntry but misses every permutation, it logs which fields differ from the closest existing permutation: static_state word index, specialization buffer or image field, or push_data_start. When a pipeline key misses, it logs whether only static_params, vertex_input or rendering differ from the previous pipeline with the same shader ids. Keep the per-stage counters, but print them to the log at most once per second and stop recounting all programs on every compile (:379-391).

**Why.** There are no zones in the shader or pipeline path (grep KYTY_PROFILER: only pipelineCache.cpp:552, 789 and 956). The 12% cannot be attributed to our frontend, SPIR-V emission, spirv-val or the driver. Several permutation multipliers exist in the code (see the next proposal), and a diff log shows which of them fire in Astro Bot. The unconditional std::printf on Thread_Gpu is also a small synchronous console write per compile on Windows.

**Impact.** No direct fps gain. It decides whether the next steps should target the frontend, the driver compile, or permutation churn, and it shows whether steady-state gameplay keeps creating permutations, which would mean an explosion rather than a cold cache.

**Risks.** Nearly none. Zones add about 20-50 ns each when Tracy is off and must stay out of per-instruction loops. The diff logger should only run when the flag is set.

**Validation.** Tracy capture on a fixed Astro Bot route (the doc's methodology): compare total time in the new zones with the ~12% sampling estimate, and count new permutations per minute after the first minute in an area already visited.

### Proposal (medium): Remove state-driven permutation multipliers: push-data placement, clip-space constants, depth bounds, core dynamic state

**What.** (a) Push data: GetGraphicsPrograms compiles the PS first and passes the advanced cursor to the VS (pipelineCache.cpp:753-761), and a permutation matches only if push_data_start_dword is equal (:304-307). Give each stage a fixed window instead: VS/LS/MS at [base,16) and PS at [16,32), with mesh keeping its 7 reserved dwords. Update AllocateBindings (BindingLayout.cpp:72-147) and the push writes in descriptors.cpp. Data that does not fit already falls back to the ShaderData binding (BindingLayout.cpp:142-144). (b) Clip-disabled draws: take clip_space.scale and offset out of the VS static key (shader.cpp:661-672) and out of the SPIR-V constants (spirvEmitterFlow.cpp:388-416). Load them from 4 shader-data dwords written per draw; half_extent depends only on the device and can stay constant. (c) Depth bounds: zero depth_min_bounds and depth_max_bounds in the key when the test is disabled (pipelineCache.cpp:886-888). Better, add VK_DYNAMIC_STATE_DEPTH_BOUNDS and DEPTH_BOUNDS_TEST_ENABLE (core 1.0 and 1.3) and set them in SetGraphicsDynamicParams, skipping this on MoltenVK as today. (d) Make CULL_MODE, FRONT_FACE, PRIMITIVE_RESTART_ENABLE and PRIMITIVE_TOPOLOGY dynamic (all core in 1.3, no extension). Keep only the topology class (point, line, triangle, patch) in PipelineStaticParameters, and keep the existing rect-list cull suppression (:889-891). Only if the driver's EDS3 feature bits allow it, also make colorWriteMask, blend enable and blend equation dynamic.

**Why.** (a) Because of the ordering at :756-761, one VS gets a different permutation for every distinct PS shader-data size. A depth or shadow pass without a PS starts at 0, while main passes start at the PS size. Each such permutation re-runs the full TranslateProgram (:356) plus emission and validation. (b) Clip-disabled viewport floats are baked into the SPIR-V, so every distinct viewport, such as a dynamic-resolution step or a mip-chain pass, produces a new VS permutation and a new pipeline. (c) The key holds DB_DEPTH_BOUNDS_MIN/MAX (depthRenderTarget.cpp:319-320) even when the test is off, so any change to those registers creates a new VkPipeline. (d) The dynamic-state list (shaders.cpp:485-508) omits cull, front face, topology, restart and depth bounds, so two-sided materials and strip/list changes multiply the pipeline count.

**Impact.** Game-dependent. The next step is to measure with the diff logger from the previous proposal. (a) should cut VS translate and emit work by roughly the number of distinct partner data sizes, typically 2-3x for VSs shared between depth and color passes, and the journal shrinks by the same amount. (b) and (c) can remove an unbounded, continuing stream of recompiles if the game changes viewports under dynamic resolution or writes depth-bounds registers per draw; that stream would be the most likely cause of steady-state translation time. (d) typically removes 1.5-3x of pipeline variants. It also helps any later async or GPL work, because fewer, more reusable pipelines means fewer skipped draws.

**Risks.** (a) Exceeding a 16-dword window moves that stage to the ShaderData UBO path, which adds a descriptor and a stream write per draw; measure how often that happens. Tessellation stages must share the VS window correctly, and mesh's 7 reserved dwords must stay intact. (b) Getting the NDC math wrong misplaces every clip-disabled draw, so compare against the current constant path in tests. (c)/(d) Setting dynamic state is mandatory once declared, so a missing vkCmdSet* before a draw is undefined behaviour and can render garbage. Rect lists (ePatchList) and tessellation must keep the patch topology class. Polygon mode stays static. On MoltenVK, keep the current path behind the existing __APPLE__ guards.

**Validation.** Existing GPU tests (attachment feedback, shader_precompile_gpu, mesh_dispatch) plus new render tests covering cull front/back/none, face winding, depth-bounds on and off with varying bounds, a clip-disabled rect draw at 3 viewport sizes, and one VS drawn with no PS, a 4-dword PS and a 12-dword PS. Assert the permutation and pipeline counts: one VS permutation and one pipeline where the state is now dynamic. Pixel-compare against the current build. In game, the permutation counter should stop rising in steady state.

### Proposal (medium): Parallel, non-serialized journal replay across all cores

**What.** Rewrite PipelineCache::ReplayPrecompiled (pipelineCache.cpp:551-622). Group records by ProgramKey, then run the groups on a worker pool of hardware_concurrency()-1 threads (15 on a 7800X3D) at below-normal priority. Each worker runs TranslateProgram, its consistency checks (:604-609), ExtractResourcePlan for the group's first record, CompileProgram, optional validation and CompileSPV without holding m_mutex. It takes the lock only to dedupe against existing permutations, insert or emplace the SourceEntry, assign ++next_shader_id, and push the permutation. Store permutations in a std::deque or as unique_ptr so the pointers the GPU thread holds stay stable. Optional second step: stop blocking the first lookup (WaitForPrecompile at :692 and :768) on the whole replay. Instead keep a map from ProgramKey to in-flight state. A live lookup of a key that is still queued moves that group to the front and waits only for it; any other key proceeds normally.

**Why.** Replay is one std::jthread (:540-541), and it holds m_mutex through translation and compilation (:589-619). Every GetGraphicsPrograms and GetComputeProgram call first joins it (:544-549), so the first draw waits for the serial sum of every recorded permutation's compile time. The recompiler has no mutable globals: grep finds only atomic warn-once flags such as ShaderRecompiler.cpp:530 and spirvEmitterFlow.cpp:536. ResourcePlan scratch belongs to each Program. vkCreateShaderModule is thread-safe.

**Impact.** Warm startup, or time to the first frame, shrinks by about 8-12x on an 8C/16T part. For example, 3,000 records at about 5 ms each goes from about 15 s to under 2 s. No steady-state fps change, because replay finishes before gameplay either way, but it makes a large journal affordable, which the cache proposal above depends on.

**Risks.** Races if any insertion or dedupe happens outside the lock, and duplicate compiles if dedupe runs only before compiling (acceptable, but wasted). Permutation pointers held by in-flight draws become dangling if a vector reallocates, hence the stable storage. Shader ids become order-dependent, which is fine because they are only in-memory pipeline keys. Log lines interleave. At full width, the replay can starve guest threads during game boot, hence below-normal priority and a hardware_concurrency()-1 cap. If a record triggers an EXIT inside the recompiler on a worker, the whole process still aborts, which is today's behaviour as well.

**Validation.** shader_precompile_gpu equivalence tests with N workers, including a TSan build on Linux. A synthetic journal of 2,000 records from the test shader generators: compare wall time for 1 thread and N threads. Check that the 'replayed X; skipped Y' counts and the set of (key, specialization, push_start) entries are identical to the serial replay.

### Proposal (medium): Record pipelines in the journal and pre-create VkPipelines in parallel at startup

**What.** Add a second record type, version 3, that stores a graphics pipeline as PipelineRenderingState, PipelineVertexInputState, PipelineStaticParameters and topology, plus the journal indices of its 1-3 vertex permutations and its PS permutation instead of runtime shader ids. It also stores the vertex_fetch_components and resources/resources_dst shape that CreatePipelineInternal reads (shaders.cpp:282-305). After the parallel shader replay, resolve the indices to ShaderProgram handles and call CreatePipelineInternal from the same worker pool, passing the shared VkPipelineCache; creation there is thread-safe and the cache is internally synchronized. Insert the results into m_graphics_pipelines under m_mutex. Do the same for compute by keying on the permutation. Append a pipeline record when GetGraphicsPipeline misses (:924-947).

**Why.** Today replay only creates shader modules. Every VkPipeline is still created lazily on Thread_Gpu at first use (:940-941), paying at least a cache-hit creation (hash, lookup, deserialize, upload), and a full compile when the driver cache is cold or was lost in a crash. Descriptor set layouts and pipeline layouts are also created per pipeline on the GPU thread (shaders.cpp:190-207, 432-470).

**Impact.** Removes first-use pipeline creation from Thread_Gpu on warm runs. That is about 0.1-1 ms per pipeline with a warm driver cache, repeated each session and showing up as micro-stutter in newly visited areas, and 10-100 ms per pipeline with a cold one. Startup cost is amortized across 15 threads.

**Risks.** Records must not persist runtime-only data such as addresses, and GetInputFormat reads vs_input_info.resources[].Format(), which must round-trip. The attachment-feedback flags and dynamic-feedback normalization must match live keys exactly, or pre-created pipelines are never hit and only waste compile time. Pipeline counts can be in the tens of thousands if the permutation fixes above are not done first. Memory for unused pipelines needs a replay cap or an LRU.

**Validation.** A GPU test that records a draw, restarts, and checks that GetGraphicsPipeline hits the pre-created pipeline, meaning no createGraphicsPipelines zone appears on Thread_Gpu. Compare Tracy traces across two runs of the same route.

### Proposal (large): Opt-in async shader and pipeline compilation with skip-draw (compute stays synchronous)

**What.** Add --async-shaders, default off. Create a CompilePool of 3-4 below-normal-priority workers during gameplay. Phase 1 (pipelines only): enable pipelineCreationCacheControl (core in 1.3; RequiredVulkan13Features at vulkanWindow.cpp:61-66 only enables dynamicRendering and synchronization2). In GetGraphicsPipeline, first try createGraphicsPipelines with FAIL_ON_PIPELINE_COMPILE_REQUIRED. On VK_PIPELINE_COMPILE_REQUIRED, deep-copy the inputs: key, rendering, vertex_input, static_params, shader module handles, and stable pointers to the CompiledShaderInfo, or copies of them. Queue the job and return nullptr. The draw path then aborts before BeginRendering (renderDraw.cpp:1102-1106); bindings were prepared but not committed. Workers publish into a lock-free completion queue that the GPU thread drains at the next lookup, keeping a single writer for m_graphics_pipelines. Phase 2 (programs): on a permutation miss for VS or PS, the GPU thread has already materialized the specialization, and it must, because TryReadGpuCleanBacking fails off the GPU thread (memory.cpp:883). It hands params.code copied, options, specialization and push start to a worker that runs TranslateProgram, CompileProgram and CompileSPV. The draw is skipped through PrepareDrawRenderState returning false (renderDraw.cpp:898-934). For a brand-new source, the worker also returns the ResourcePlan, and materialization happens on the draw's next occurrence. Never skip: compute dispatches; draws whose translated stages have HasShaderMemoryWrites (ShaderIR.h:600) or uses_dma; and draws while a user-configurable 'sync budget' (for example up to 2 ms of waiting per frame) has not been spent. When a skip is not allowed, fall back to a blocking wait.

**Why.** Every compile step runs synchronously on the bottleneck thread: TranslateProgram (pipelineCache.cpp:356), CompileProgram, spirv-val and CompileSPV (:246-256), and vkCreateGraphicsPipelines (shaders.cpp:550). Thread_Gpu is at 50-79% of a core with the GPU only 40% busy. Pipeline creation is thread-safe in Vulkan, and the recompiler is reentrant.

**Impact.** Removes whatever compile time remains after the cache and permutation work, up to the full ~12% plus multi-frame hitches, from Thread_Gpu in first-seen areas. Frame pacing improves more than average fps. Phase 1 alone covers the driver-compile part at medium effort.

**Risks.** Visual problems. Objects pop in for 1-N frames. Temporal effects (TAA, SSR history, exposure) smear. Render-to-texture passes that run only once, such as a LUT, font or decal atlas, a probe capture, or an impostor bake, keep permanent garbage because the draw never repeats. Depth written by a skipped prepass can make GPU-driven culling hide geometry for a frame. Game logic can diverge if a skipped draw feeds a later CPU readback, which is why draws with memory writes and all compute stay synchronous. Engineering risks: the permutation vector's pointers (input_info.stage.program at pipelineCache.cpp:310-311 and 372) must be stable storage; worker lifetime must be joined before the device, the driver cache or the ProgramCache is destroyed; and CompileSPV's EXIT-on-failure paths on workers still abort the process. PR #35 shows the wrong way: drawing with empty fallback shaders writes undefined color and depth into real targets.

**Validation.** A unit test with a gate-controlled mock compile: a draw is skipped while its pipeline is pending and emitted after the pipeline completes, and compute or writing draws never skip. A TSan run of the GPU tests with async enabled and forced COMPILE_REQUIRED. In game, compare frame-time percentiles and the count of skipped draws per second with and without the flag on a cold cache, and inspect screenshots for persistent corruption.

### Proposal (small): Cheaper per-draw lookup keys, and a fast path for 'same as the previous draw'

**What.** (1) Pipeline key: make GraphicsPipelineKey a packed POD with explicit zeroing (padding in Binding and Attribute is currently unspecified, pipelineCache.h:121-137) and hash it with a single XXH3_64bits over the whole struct, instead of 129 per-byte Mix calls plus per-field mixes (:198-233). (2) Keep last_key and a last_pipeline pointer in PipelineCache; memcmp the new key against last_key and skip the unordered_map when they are equal, which is common for consecutive draws. (3) Program lookup: compute the ProgramKey hash while building the static key, fold it into ProgramKeyHash, and keep a small cache of the last few (shader base address, static-key hash) pairs mapped to SourceEntry*, so repeated draws skip the find and the 60-430-word vector comparison (shader.cpp:653-759; pipelineCache.cpp:283-288). (4) Keep the 2-3 most recently hit permutations at the front of SourceEntry::permutations.

**Why.** Every draw rebuilds a ~720-byte key and walks a serial hash chain of more than 150 dependent Mix steps, while equality compares all 32 bindings and 32 attributes. Every stage rebuilds a static-key vector of up to 13+32*13 words (MaxStaticKeyWords, :228). The key cannot be hashed as raw bytes today because the padding is unspecified.

**Impact.** Roughly 0.2-0.5 µs per draw. At an assumed 2-5k draws per frame, that frees about 0.5-2 ms per frame on Thread_Gpu, or 1-5% of a 37 ms frame. Low risk, independent of the other proposals.

**Risks.** Hashing raw bytes with nonzero padding produces spurious misses, which only wastes duplicate pipelines, or, if equality were ever changed to memcmp without zeroing, spurious mismatches. The last-key memo must be invalidated whenever pipelines are destroyed. Any key change must stay consistent with the journal records described above.

**Validation.** A microbenchmark in the tests: 10^6 key builds and lookups, before and after. Assert that the hash is stable across value-initialized keys. Compare Tracy zone time per draw for GetGraphicsPipeline and ProgramCache::Get on the Astro Bot route.

### Proposal (small): Turn off spirv-val by default in the launcher, and trim other synchronous extras on the compile path

**What.** Change the launcher default shader_validation_enabled from true to false (src/launcher/include/configuration.h:99). The CLI default is already false (emulatorConfig.h:58), but mainDialog.cpp:235 always passes the launcher value. Where validation is wanted, run it on a worker after the module has been created and log the failure rather than blocking the draw, keeping EXIT only in debug. Also: remove the per-compile std::printf and the O(programs) recount (pipelineCache.cpp:379-391); cache BuildRectListShaders modules per (VS output signature, PS input signature) instead of regenerating and destroying them for every rect-list pipeline (shaders.cpp:233-246, 552-557); and move Journal::Append's write and flush (shaderPrecompile.cpp:409-432) to the background writer.

**Why.** ValidateShaderSpirv builds a SpirvTools context and fully validates every new permutation on Thread_Gpu (pipelineCache.cpp:150-178, 249). Launcher users therefore pay a whole-module validation pass, often comparable to or larger than emission, on every compile. The launcher also defaults to shader_optimization_type Performance, which only changes the cache signature, not code generation.

**Impact.** For users who start through the launcher, this probably removes a large part of the per-compile recompiler cost on cold caches; how much should be measured with the zones from the instrumentation proposal. CLI users see no change. The other items are small, about 0.1-1 ms per compile.

**Risks.** With validation off, invalid SPIR-V reaches the driver, which can crash or misbehave instead of stopping cleanly with a diagnostic. Keep it on in CI and in the test harness, and document the flag for bug reports.

**Validation.** Compare time in the ValidateShaderSpirv zone with and without --shader-validation on a cold-cache run. Keep CI running the full shader suite with validation on.

### Proposal (medium): Stop re-running the whole frontend for each new permutation of a cached source

**What.** Store the decode and CFG results for each source in SourceEntry: Decoder::Program, joined_code, the structurized CFG::Graph and the embedded-fetch plan, all pure functions of code, back_code, stage and static inputs. A permutation miss then reruns only Frontend::TranslateProgram and the IR passes (ShaderRecompiler.cpp:585-628). A larger step is to implement IR::Program::Clone() with value remapping, keep one post-TrackResources program per source, and clone it before ApplyResourceSpecialization. Separately, when a source has seen more than K distinct buffer packed_stride values, compile a runtime-stride variant: specialize with packed_stride = 0 plus a flag, and have BufferByteAddress (spirvEmitterMemory.cpp:83-96) read the stride from the descriptor dwords pushed through shader data.

**Why.** When an entry exists but no permutation matches, pipelineCache.cpp:356 still calls TranslateProgram from scratch: decode, CFG build and structurize (ShaderCFG.cpp is 2,313 lines), IR, SSA and all the passes. Only ApplyResourceSpecialization, CollectShaderInfo, AllocateBindings and EmitProgram actually depend on the permutation (:630-647). Buffer stride, format and swizzle enter ResourceSpecialization (ResourceMaterialization.cpp:421-450) and are emitted as SPIR-V constants, so generic kernels bound to buffers of different strides fork a permutation each time.

**Impact.** Depends on the phase split and on how many permutations each source has, which the instrumentation proposal will show. If decode and CFG are about half of the frontend and sources average 2-3 permutations, recompiler time drops by about 25-40%. Adaptive stride removes an open-ended source of compute permutations.

**Risks.** Memory: cached decode/CFG per source, or a full Program. Cap it or drop it after the first N permutations. A buggy IR clone produces miscompiles that are hard to diagnose. A runtime-stride shader is slightly slower on the GPU (cheap at 40% GPU load) and must reproduce the swizzle and ADD_TID address math exactly.

**Validation.** Byte-compare the SPIR-V from the cached path and the fresh path for every test shader and specialization. Run the full shader suite and the stride and swizzle buffer tests. Compare the phase zones before and after.

### Proposal (very large): Defer VK_EXT_graphics_pipeline_library and VK_EXT_shader_object until pipeline layouts are shared

**What.** Before any GPL work, move from per-pipeline descriptor layouts (shaders.cpp:190-207, 432-470) to a shared layout family. The binding numbers are already fixed per stage group and kind (ShaderIR.h:313-320), so the missing piece is fixed descriptor counts per binding (partially bound) or VK_EXT_descriptor_buffer, because push descriptors allow only one push set per layout and are capped by maxPushDescriptors. Then build four library types: vertex-input libraries keyed by PipelineVertexInputState plus formats; pre-raster libraries keyed by VS permutation plus now-dynamic rasterizer state; fragment-shader libraries keyed by PS permutation, sample shading and depth state; and fragment-output libraries keyed by rendering formats, blend and samples. Fast-link on Thread_Gpu, and build the link-time-optimized pipeline on the compile pool, swapping it in when ready. That is DXVK's model, and it needs no skipped draws. Treat shader objects as a later option: they also need VK_EXT_vertex_input_dynamic_state and EDS3 for blend, mask, samples, polygon mode and depth clip. Check support on the RX 9070 XT's Windows driver with vulkaninfo; RADV exposes both extensions.

**Why.** Each VS/PS/state combination is a full monolithic compile (shaders.cpp:550). Cheap linking and reuse across state variants is exactly what GPL is for. It is blocked today because the layout depends on both stages' binding counts (AddLayoutBindings for VS and PS into one set, shaders.cpp:435-448), so a VS library could not be reused with a different PS.

**Impact.** Potentially the best long-term fix for pipeline compile cost without the visual risks of skipped draws. On its own it does not reduce our recompiler (TranslateProgram) time.

**Risks.** A descriptor-model rewrite touches every bind path (descriptors.cpp) and the MoltenVK path. Fast-linked pipelines run slower on the GPU until the optimized one arrives. Driver bugs are possible in less-used GPL paths.

**Validation.** Only after the layout refactor passes the full GPU suite: A/B pipeline creation time in Tracy, GPU frame time with fast-linked versus optimized pipelines, and the complete render regression tests.

### Facts

- All shader and pipeline creation is synchronous on Thread_Gpu: ProgramCache::Get calls TranslateProgram (pipelineCache.cpp:356) then CompilePermutation, which runs CompileProgram, optional ValidateShaderSpirv, CompileSPV/vkCreateShaderModule (:246-256), then Journal::Append with flush (:373-376, shaderPrecompile.cpp:424). GetGraphicsPipeline runs CreatePipelineInternal, which calls vkCreateGraphicsPipelines (pipelineCache.cpp:940-941, shaders.cpp:550). Only startup journal replay uses another thread, and the first lookup blocks on it (WaitForPrecompile, pipelineCache.cpp:544-549, 692, 768).
- A new permutation of an already-cached source re-runs the entire frontend: decode, CFG build/structurize, IR, SSA, constant propagation, DCE, ReadLane, BuildSrtPlan and TrackResources (pipelineCache.cpp:356; ShaderRecompiler.cpp:486-628). Only CompileProgram (:630-668) depends on the specialization.
- Per-draw cached path: RefreshShaders (renderDraw.cpp:867-895) calls GetGraphicsPrograms, which runs PrepareProgram (global g_shader_map_mutex lookup, shader.cpp:81-91; semantics parse), BuildStageStaticKey (up to 13+32*13 words, pipelineCache.cpp:228) and unordered_map find. It then runs MaterializeResources, a two-walker SRT evaluation, on every draw (pipelineCache.cpp:298-300; ResourceMaterialization.cpp:932-1060), followed by a linear permutation search (:301-309). GetGraphicsPipeline then builds a ~720-byte key hashed with 129 per-byte Mix calls (pipelineCache.h:198-233).
- The VS push-data start depends on the PS: the PS is looked up first and advances push_data_cursor before the VS lookup (pipelineCache.cpp:753-761). Permutations match only on equal push_data_start_dword (:304-307), so one VS paired with PSs of different shader-data sizes (or with no PS) compiles multiple times.
- When clip_disable is set, the viewport scale and offset floats go into the VS static key (pipelineCache.cpp:738-751; shader.cpp:661-672) and are emitted as SPIR-V constants (spirvEmitterFlow.cpp:388-416). Each distinct viewport therefore creates a new VS permutation and pipeline.
- Depth-bounds min/max are in the pipeline key even when the depth-bounds test is disabled (depthRenderTarget.cpp:318-320; pipelineCache.cpp:886-888). Cull mode, front face, topology, primitive restart and depth bounds are static pipeline state; the dynamic-state list (shaders.cpp:485-508) omits them, although they are core in Vulkan 1.3.
- Buffer packed_stride, format and swizzle are part of ResourceSpecialization (ResourceMaterialization.cpp:421-450) and are baked as SPIR-V constants (spirvEmitterMemory.cpp:83-96).
- Cache gating (pipelineCache.cpp:438-456, 524-529) disables both the VkPipelineCache and the shader journal unless all of these hold: CMAKE_BUILD_TYPE is not Debug (CMakeLists.txt:91-95), the git hash is known and not '-dirty' (generate_version.cmake:30-38), TITLE_ID/CONTENT_ID exists, and --shader-precompile is true. The signature includes the full KYTY_GIT_REVISION, driver identity, pipelineCacheUUID and the opt type (:64-74); the journal key adds APP_VER (:532-535). Files are _PipelineCache/<TITLE_ID>.bin and .shaders.
- The VkPipelineCache is saved only at clean WindowRun exit (window.cpp:820) and destroyed after saving (pipelineCache.cpp:683-684), so a crash loses the session's driver cache. The journal flushes each record and survives crashes, but it holds shader permutations only, not pipelines, and caps at 32,768 records (shaderPrecompile.cpp:19).
- Journal replay is a single jthread (pipelineCache.cpp:540-541) that holds m_mutex across TranslateProgram and CompilePermutation for each record (:589-619). It compiles with zero user-data values, which is by design: only the SGPR count affects code.
- The recompiler has no mutable global state beyond atomic warn-once flags, so it is safe to run on multiple threads. Materialization is GPU-thread-only: TryReadGpuCleanBacking returns false when !IsGpuThread() (memory.cpp:881-889), and ResourcePlan evaluation scratch is documented as GPU-thread scratch (ShaderIR.h:562).
- The launcher defaults shader_validation_enabled=true (src/launcher/include/configuration.h:99) and always passes --shader-validation (mainDialog.cpp:235); the CLI default is false (emulatorConfig.h:58). With validation on, every new permutation runs spvtools Validate synchronously (pipelineCache.cpp:150-178).
- Each new permutation triggers an unconditional std::printf on Thread_Gpu and an O(number of programs) recount (pipelineCache.cpp:379-391). Rect-list pipelines regenerate and create/destroy their TCS/TES modules on every pipeline creation (shaders.cpp:233-246, 552-557). Each pipeline gets its own descriptor set layout and pipeline layout (shaders.cpp:190-207, 432-470).
- There are no Tracy zones around TranslateProgram, CompileProgram, validation or vkCreate* (grep KYTY_PROFILER finds only pipelineCache.cpp:552, 789, 956), so the ~12% figure cannot currently be split by phase. The zone at :789 also includes the per-draw lookup.
- pipelineCreationCacheControl is not enabled (RequiredVulkan13Features, vulkanWindow.cpp:61-66), so VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT is not usable yet. The branch does not use VK_EXT_graphics_pipeline_library, VK_EXT_shader_object or EDS3.
- PR #35 (pr/35, based on c21a4100, July 2026) targets the old render-pass architecture (renderer/pipelineCache.cpp, VkRenderPass framebuffers) and does not apply to this branch. It is also unsound. Its 'fallback' pipelines are compiled synchronously per key (GetOrCreateDynamicFallbackPipeline calls CreatePipelineInternal), so no time is saved. Their SPIR-V is empty: the VS writes no Position and the FS writes no outputs, while the pipeline keeps real depth/color write state, so undefined values are written into real targets. The renderDraw nullptr 'skip' never fires because a fallback is always returned. Workers capture raw VkRenderPass handles, spawn hardware_concurrency() threads, and write the cache file non-atomically without a checksum. The PR also commits test_shader.* artifacts and duplicate headers. Only the idea of periodically saving the cache is worth porting; per-game cache files already exist on this branch.
- No upstream open PR other than #35 targets shader-compile or pipeline-creation performance (scan of open-prs.txt). A GitHub issue search for shader/pipeline stutter found only #281 (missing opcodes).

## Threading model and multi-core use (Performance survey)

The emulator runs guest x86-64 code natively. Each guest pthread is its own host thread (pthread.cpp:3286 calls host pthread_create, then RunThread and RunOnGuestStack at 3218). Guest affinity is stored but never applied (pthread.cpp:2075-2084). On Windows, guest priority maps to host ±2, so guest priority ≤478 becomes THREAD_PRIORITY_HIGHEST (pthread.cpp:2160-2170, 3499-3505). On Linux it is ignored. Guest mutexes, condition variables, semaphores and event queues are per-object std::mutex or CondVar, polling for signals every 10 ms. Their fast paths take no global lock. The guest CPU side is therefore already multi-core.

The host service threads are:
- **Main thread:** the SDL event loop (emulator.cpp:171-181, window.cpp:709-729).
- **Guest main thread:** runs RuntimeLinker::Execute (emulator.cpp:173).
- **Thread_Gpu** (graphicsRun.cpp:87, 469-560): one consumer for the graphics queue, all 56 compute queues and synchronous fault/unmap commands. It parses PM4, tracks state, translates shaders and creates pipelines, does buffer/texture cache work, records Vulkan commands and calls vkQueueSubmit.
- **Priority-operations thread** of CommandScheduler (commandScheduler.cpp:100-104, 244-270): runs EOP interrupts, flip completion and readback publication after timeline waits. A second instance belongs to the presenter's scheduler (swapchain.cpp:317, 354).
- **VideoOut present thread** (videoOut.cpp:251, 801-872): generates vblank and also presents.
- **ShaderPrecompile thread:** startup only (pipelineCache.cpp:540).
- **Minor threads:** the detached-thread reaper (every 10 s), the audio3d playback thread, SDL's audio device thread, AvPlayer workers (guest pthreads) and one std::thread per signal. There is no file-IO thread; IO runs on the calling guest thread.

Single-threaded chokepoints and big locks:
- Thread_Gpu itself.
- **RenderContext::m_mutex:** held for the whole of every draw or dispatch, including shader compiles and drains inside resource preparation. The present path takes it too (swapchain.cpp:725, 748, 798).
- **TextureCache::m_lock:** a pure spin lock, taken by every CPU write fault on a tracked page.
- **g_memory_operation_mutex:** one recursive lock for all memory operations, held across GPU-unmap drains.
- **GuestGpu::m_submission_mutex:** held while AgcSuspendPoint waits for Thread_Gpu to go idle.
- **libUlt g_ult_mutex:** taken on every ULT mutex lock and unlock.

Can we have multi-core? The guest CPU already has it. Host GPU emulation is single-threaded by design. The caches require GPU-thread serialization (bufferCache.h:66-67). PM4 handlers perform guest-memory side effects at parse time: EOP label memcpy, WRITE_DATA, WAIT_REG_MEM reads, indirect-argument and predicate reads, and 8-bit index expansion. A PM4 front-end/back-end split or parallel secondary command buffers would therefore run in lockstep.

The real multi-core wins, ranked:
1. **Stop serializing the guest's frame building with Thread_Gpu at AgcSuspendPoint.** Today Done() waits for Thread_Gpu to go fully idle. Replace this with bounded pipelining.
2. **Fix host thread priorities and guest spin-sleeps**, which can starve Thread_Gpu and the priority thread.
3. **Move vkQueueSubmit and vkQueuePresentKHR to a queue-owner thread**, and later all vkCmd recording (Yuzu-style).
4. **Decouple vblank and present from the render mutex.**
5. **Parallelize shader/pipeline compilation:** multi-worker journal replay, journaled pipeline pre-creation, and concurrent VS/PS permutation compiles.

Timing and lock hygiene are cheap wins that make frame times steadier: sub-millisecond CondVar waits on Windows, and spin-then-park locks on fault paths. A second VkQueue for async compute gives nothing while the host GPU is only 40% busy.

### Proposal (small): 1. Let guest frame building overlap Thread_Gpu at AgcSuspendPoint (bounded frames in flight)

**What.** Change GuestGpu::Done (graphicsRun.cpp:199-207), which today calls WaitForIdle. Use an epoch scheme instead:
- Done() increments m_issued_epoch and enqueues a zero-work marker Submission (new SubmissionType::SuspendPoint, queue 0). When Process() reaches the marker it stores m_processed_epoch and signals a new CondVar.
- Done() returns immediately unless m_issued_epoch - m_processed_epoch > K. K comes from a new config/CLI option, --gpu-frames-in-flight, where 0 keeps today's behavior.
- Keep setting m_graphics_done = true in Done() so the next Submit() still captures reset_processor. Queue 0 is FIFO, so the CommandProcessor::Reset lands between the same packets as today.
- Do the bounded wait after releasing GpuMutexLock(m_submission_mutex), so other guest threads can keep submitting ACBs.

Step 0: add a Tracy zone and counters around the wait to measure how long the guest render thread blocks per frame.

**Why.** - Done() is the whole implementation of sceAgcSuspendPoint (agc.cpp:1551-1557). WaitForIdle (graphicsRun.cpp:461-466) blocks until Thread_Gpu has no queued or processing work, including any GPU drains it hits. It holds m_submission_mutex the whole time.
- So the guest render thread cannot build frame N+1 while Thread_Gpu processes frame N. Frame time becomes T_guest + T_gpu_thread instead of max(T_guest, T_gpu_thread).
- This fits the profile: Thread_Gpu is only 50-79% busy (idle while the guest builds) and the GPU is about 40% busy.
- Guest-visible EOP labels are already written at parse time (graphicsRun.cpp:1170-1175, memcpy before recording). Label-gated reuse of command memory is therefore already safe with Thread_Gpu running ahead.
- A 1-slot queued version (commit 90bf3c92) landed on 2026-08-07 and was reverted the same evening (ec78afb9) with no reason recorded. Ask the author (nmzik) before re-landing.

**Impact.** Potentially the largest single FPS lever in this area:
- If Astro Bot calls sceAgcSuspendPoint once per frame and its render thread needs about 8-12 ms per frame, overlapping could move roughly 27 fps toward the Thread_Gpu-bound ceiling of 1/T_gpu_thread, around 35-40 fps. This is an estimate.
- There is no gain if the game already gates each frame on the previous frame's flip or labels. The Step 0 measurement decides.
- It also unblocks async-compute submitters that currently queue behind m_submission_mutex.

**Risks.** - **Command-memory reuse:** a guest may reuse DCB/IB memory right after SuspendPoint, relying on it as a barrier instead of on EOP labels. Thread_Gpu would then parse overwritten packets and hit EXIT on an unknown op or render wrongly.
- **Deadlock:** waiting while holding m_submission_mutex can deadlock if a blocked WAIT_REG_MEM needs another submission to make progress.
- **Semantics:** m_done_num/GetFrameNum changes meaning (logging only). Any hidden assumption that Thread_Gpu is idle after SuspendPoint breaks, such as the RenderDoc capture flow removed in bac6bf65.
- Default K=0 until validated on the tested-games list.

**Validation.** - A Tracy trace before and after: the render thread's time inside Done() should drop to near zero, and Thread_Gpu's idle gaps should shrink.
- A/B the fresh-frame FPS reported by the window title (not presentation rate) for K=0/1/2 on a fixed save and route.
- Add a debug mode that XXH3-hashes each DCB span at Submit() and re-hashes it when Process() first touches it, with EXIT on mismatch. This catches premature guest reuse.
- Run the full CTest suite plus a long soak run.

### Proposal (small): 2. Raise host-thread priorities and make guest spin-sleeps yield the core

**What.** (a) Windows priorities:
- Set THREAD_PRIORITY_HIGHEST on Thread_Gpu (start of ThreadRun, graphicsRun.cpp:469), on both CommandScheduler priority threads (PriorityOperationsThread, commandScheduler.cpp:244) and on the VideoOut present/vblank thread (videoOut.cpp:801).
- Alternatively, cap the guest mapping at +1 (ABOVE_NORMAL) at pthread.cpp:2160-2170 and 3499-3505.
- On Linux use setpriority/nice for the same threads.

(b) Optional experiment: reserve one physical core, both SMT siblings, for Thread_Gpu by pinning it there and applying a host affinity mask that excludes that core to guest threads in RunThread. Guest affinity is currently ignored anyway.

(c) Guest sleeps: KernelUsleep and KernelNanosleep (pthread.cpp:3844-3887) route through Common::Thread::SleepMicro/SleepNano. For waits of 50 µs or less these busy-spin with QPC + YieldProcessor (threads.cpp:43-60) or clock_gettime (threads.cpp:135-160). Add a guest-sleep variant that spins at most a few µs, then loops on SwitchToThread (Windows) or sched_yield (Linux) until the deadline.

**Why.** - Guest threads at guest priority ≤478 run at THREAD_PRIORITY_HIGHEST on Windows (pthread.cpp:2160-2170). Every emulator thread is a default-priority std::jthread (graphicsRun.cpp:87, commandScheduler.cpp:100-104, videoOut.cpp:251, pipelineCache.cpp:540).
- Console job systems often keep worker threads busy with short usleep or yield polling. Here those sleeps are busy-spins, so the Windows scheduler sees the workers as always runnable at high priority.
- The priority thread is on the critical path of every drain: ReadMemory waits for the tick and then for WaitPriorityOperations (bufferCache.cpp:266-270). Any ready-queue delay for it, or for Thread_Gpu, adds directly to frame time.
- A spinning thread on Thread_Gpu's SMT sibling also takes execution resources from the bottleneck thread.

**Impact.** This mainly reduces variance (the 20-40 fps spread) and speeds up drain turnaround, because the priority thread wakes promptly. Expect 0-15% average gain, depending on how many runnable high-priority guest threads Astro Bot has. It is cheap to try with ETW.

**Risks.** - HIGHEST on Thread_Gpu can starve guest threads if Thread_Gpu spins. Today it mostly blocks, except in texture-cache spin sections (see proposal 8).
- Core reservation hurts games that need all 16 logical CPUs.
- Yield-based guest sleeps are less precise and could change timing-sensitive guest loops. Keep the spin for host-internal sleeps that need precision.
- Linux nice changes need CAP_SYS_NICE or rlimits.

**Validation.** - Windows Performance Recorder/WPA CPU Usage (Precise): measure ready time of Thread_Gpu and the priority thread before and after.
- Tracy: context-switch capture.
- Frame-time percentiles (p50/p99) on a fixed route. Also count guest threads by host priority at runtime (add a log in PthreadAttrSetschedparam).

### Proposal (medium): 3. Queue-owner submit thread: move vkQueueSubmit/vkQueuePresentKHR off Thread_Gpu

**What.** Add a SubmitWorker thread that exclusively owns graphics.queue.
- CommandScheduler::Submit (commandScheduler.cpp:333-379) keeps m_command.End(), which must stay on the recording thread because the VkCommandPool is externally synchronized. It also keeps tick = m_master.NextTick().
- Submit then pushes {cmdbuf, SubmitInfo with the timeline signal, tick, debug args} into a FIFO and returns. The worker calls vkQueueSubmit in FIFO order and reports failures with the same diagnostics.
- Route the presenter scheduler's Submit and Swapchain::Present (swapchain.cpp:676-700) through the same FIFO, so queue order between GPU-thread and present-thread submissions is preserved. The present thread can block on a per-item completion for the present result.
- MasterSemaphore::Wait on a not-yet-submitted tick stays valid, because timeline-semaphore host waits may precede their signal submission.
- CommandPool::Commit already refuses to reuse command buffers until the GPU tick passes (commandScheduler.cpp:56-84).
- Later, move vkEndCommandBuffer too by giving the worker its own pool (see proposal 6).

**Why.** - Every Flush is a synchronous vkQueueSubmit on Thread_Gpu, and there are many:
  - after each guest submission slice (graphicsRun.cpp:612, 638)
  - after every RELEASE_MEM label with data_sel 1, and after every interrupt (pm4Handlers.cpp:2282, 2321, 2339)
  - on EOP flip (graphicsRun.cpp:1549) and CPU flip (graphicsRun.cpp:1564)
  - inside every drain (commandScheduler.cpp:176-181)
- Upstream PR #506 tried to batch RELEASE_MEM flushes because of this cost; it was not ported for other reasons.
- vkQueueSubmit through a WDDM kernel-mode driver typically costs tens of µs of CPU each (estimate, measure it). That time comes straight out of the bottleneck thread.

**Impact.** Thread_Gpu CPU drops by N_submits × submit_cost. At a plausible 50-200 submits per frame and 20-80 µs each, that is roughly 1-16 ms per frame, which is significant against a ~37 ms frame. Must be measured: add a Tracy zone and counter around queue.submit. It also makes the frequent label/interrupt flushes nearly free, so there is no need to batch them the way #506 did.

**Risks.** - **Ordering:** CPU flips rely on queue submission order. PrepareCpuFlip calls Flush and then CompleteFlip immediately (graphicsRun.cpp:1553-1567), and the present blit is submitted later from another thread. If any submit bypasses the FIFO, the blit can run before the copy, causing tearing or garbage frames.
- Swapchain recreation (swapchain.cpp:503, waitIdle) and shutdown must drain the FIFO first.
- Device-lost diagnostics (m_debug_*) must travel with each item.
- Holding queue_mutex is no longer enough for external code such as RenderDoc or the system overlay if they submit directly.

**Validation.** - Tracy counters: submits per frame and time in vkQueueSubmit on Thread_Gpu, before and after.
- Enable Vulkan validation plus synchronization validation in a debug build and run the GPU CTest suites (pending-writeback races, shader_precompile_gpu).
- A/B fresh-frame FPS; check flip ordering visually and with frame-dump captures.

### Proposal (small): 4. Decouple vblank generation from presentation, and get present off RenderContext::m_mutex

**What.** (a) Split VideoOutDriver::Impl::PresentThread (videoOut.cpp:801-872) into two threads:
- A high-priority vblank timer thread that only runs VblankBegin/VblankEnd (videoOut.cpp:762-799) on schedule. It takes only m_mutex and the per-cfg mutexes.
- A present thread that waits on m_submit_cond_var for Ready flips, keeping flip-due checks keyed to the vblank counter.

(b) Audit why Presenter::Present (swapchain.cpp:781-815) and PrepareBlankFrame (748) take RenderContext::m_mutex. They record only into the presenter's own CommandScheduler and presenter-owned frame and swapchain images. Frame hand-off is already ordered by FlipQueue and FramePool mutexes. Remove or narrow the lock, keeping it only where the texture cache is touched (PrepareFrame's ResolveSurface runs on Thread_Gpu anyway).

**Why.** Thread_Gpu holds RenderContext::m_mutex for the whole of every draw and dispatch: renderDraw.cpp:1203 and 1314, renderCompute.cpp:221 and 412, graphicsRun.cpp:1398. Inside that scope it can:
- compile shaders and pipelines (RefreshShaders reaches PipelineCache::GetGraphicsPrograms and CreatePipelineInternal, pipelineCache.cpp:627-760 and 940)
- wait for the whole startup precompile replay (WaitForPrecompile at pipelineCache.cpp:627)
- submit and wait during resource preparation (comment at renderDraw.cpp:1101-1103)

The present thread takes the same mutex between VblankBegin (videoOut.cpp:833) and VblankEnd (869). A long draw, a drain or a shader compile on Thread_Gpu therefore delays the guest's vblank events, VideoOut vblank_cond, and presentation itself. On Windows Flip(0) also sleeps about 1 ms (see proposal 7).

**Impact.** Mostly steadier pacing: vblank intervals stop inheriting Thread_Gpu stalls. There is some FPS gain for games that pace on vblank or flip events, because they are no longer delayed by up to one drain or compile. The startup precompile no longer freezes vblank either.

**Risks.** - Flip-at-vblank semantics and the ordering of Flip vs Vblank events in TriggerVideoOutEvents must stay as today.
- RenderDoc capture (renderDoc.cpp:172) and RenderDocOnGuestFlip use the render mutex to bracket frames, so keep that path locked.
- Removing the lock without the audit could race on shared VMA or DeviceMemory objects in Frame::Configure (called from PrepareFrame on Thread_Gpu).
- Swapchain recovery (RecoverSwapchain) must still exclude concurrent queue use.

**Validation.** - Log the processTime deltas of vblank_status.count under heavy load (a shader-compile storm, a drain-heavy scene); jitter should stay under 0.5 ms.
- Tracy lock-contention capture on the render mutex.
- Run the frame_statistics test and visually check the flip order.

### Proposal (medium): 5. Parallel shader and pipeline compilation (startup replay, pipeline pre-creation, concurrent runtime permutations)

**What.** (a) ReplayPrecompiled (pipelineCache.cpp:551-619) is one thread, and it holds PipelineCache::m_mutex from the lookup (589) through TranslateProgram (601) and CompilePermutation, including vkCreateShaderModule.
- Run it as N = hardware_concurrency-2 workers.
- Each worker translates and compiles without the lock, then takes m_mutex only for the dedupe check and insert. next_shader_id must be assigned under the lock.
- Make ProgramCache::lookup_key (pipelineCache.cpp:408) per-call instead of a shared member.

(b) Journal the GraphicsPipelineKey inputs and pre-create VkPipelines on the same workers at boot, using the driver VkPipelineCache. Only shader permutations are journaled today (shaderPrecompile.h:18-29). Every pipeline is created synchronously on first use (pipelineCache.cpp:940, 972).

(c) At runtime, when one draw misses several stage permutations (VS+PS, or LS/HS/DS), compile them concurrently on the pool and join.

(d) Longer term: VK_EXT_graphics_pipeline_library, DXVK-style. Fast-link at draw time and swap in an optimized pipeline built on a background thread.

Move Journal::Append file IO (pipelineCache.cpp:365) to a background writer.

**Why.** - About 12% of Thread_Gpu time is runtime shader translation.
- Permutations are keyed on resource specialization, so new ones keep appearing during play (pipelineCache.cpp:300-370). Each miss runs TranslateProgram, CompileProgram, optional SPIR-V validation and vkCreateShaderModule inline, under the render mutex.
- vkCreateGraphicsPipelines also runs inline.
- The first draw blocks on the whole single-threaded replay (WaitForPrecompile, pipelineCache.cpp:627, 692, 768).
- The recompiler looks reentrant: the only mutable statics are atomic log-once flags, and the precompile thread already runs TranslateProgram concurrently with Thread_Gpu.

**Impact.** - **Warm boot:** replay time drops by roughly the worker count (about 6-8x on a 7800X3D).
- **Runtime:** multi-stage misses cost about as much as a single stage, removing part of the 12%. Pre-created pipelines remove first-use pipeline hitches.
- Mostly improves steadiness (fewer spikes to 20 fps) rather than the average.

**Risks.** - **Journal order:** it becomes nondeterministic, so replay must stay idempotent. It already dedupes on specialization and push-data start.
- **Shader ids:** ids are part of pipeline keys, so id assignment must stay under the lock and be stable per permutation.
- **Memory:** usage spikes during parallel compile, and SPIR-V validation (spvtools) instances must be per-thread.
- **Pre-creation:** stale journaled keys waste boot time, so cap and age them. Pipelines built with the wrong attachment-feedback flags would be unused rather than wrong.
- **GPL:** needs driver support checks. The RX 9070 XT Windows driver's GPL/fast-link support must be verified.

**Validation.** - shader_precompile_record and shader_precompile_gpu tests, plus live-vs-replayed equivalence.
- Time the replay wall clock at boot, and use Tracy zones for compile time on Thread_Gpu during a fixed route.
- Count pipeline creations after warm boot; this should be near zero on a repeated route.

### Proposal (large): 6. Deferred Vulkan command recording on a worker thread (Yuzu-style), built on proposal 3

**What.** - Replace direct vk::CommandBuffer use with a CommandRecorder that appends commands to a chunk: POD structs or small closures with inline storage, like Yuzu's Scheduler::Record.
- Thread_Gpu keeps all resolution work and all CPU-side state tracking (Image::Transit layout state, BeginRendering state and the rest) and only enqueues commands.
- The worker owns the VkCommandPool, replays chunks into VkCommandBuffers, and ends and submits them (it can be the proposal 3 thread).
- Anything that waits for the GPU must first flush and seal the pending chunk: FlushAndWait, Wait(CurrentTick()), Finish, stream-buffer wraps.
- Migrate the roughly 55 Handle() call sites incrementally, starting with the hot ones: descriptors.cpp:1117-1131 (pushConstants, pushDescriptorSetKHR), renderDraw.cpp:1101-1170 (bindVertexBuffers2, bindIndexBuffer, dynamic state, bindPipeline, draw, drawMeshTasks), CommandBuffer::BeginRendering/EndRendering (context.cpp:63-120), image barriers, and buffer copies in bufferCache.cpp/streamBuffer.cpp.

**Why.** - Every vkCmd* executes inline on the bottleneck thread: per-draw push descriptors, barriers, dynamic rendering begin/end, pipeline binds and draws. vkBeginCommandBuffer also runs on every flush (context.cpp:31-41).
- The PM4-split alternative (proposal 10) cannot move this work, because side effects must stay ordered.
- The Vulkan API boundary is the only clean cut where the consumer never touches guest memory. renderDraw.cpp:1098-1101 already documents that after resource preparation no operation touches guest memory.

**Impact.** Moves the driver's recording CPU cost off Thread_Gpu. In similar emulators this is typically 10-25% of per-draw CPU (estimate). Measure first with ETW or Tracy on driver DLL time (amdvlk64.dll) during the draw-heavy phase. Worth it only if that share is 10% or more after proposals 1 and 3.

**Risks.** - Any call site left recording directly into a VkCommandBuffer the worker also uses is a data race. Enforce this by making Handle() private.
- Debug and crash info (SetDebugInfo) must be captured per chunk.
- Everything captured by commands must stay valid until replay: Pipeline& references, descriptor info arrays built in reused member vectors (m_descriptor_writes etc., descriptors.cpp:988-994) and push-data copies. Copy them into the chunk.
- Cross-thread latency adds to drains unless flushes wake the worker immediately.

**Validation.** - Run the full GPU CTest suite with validation and sync-validation layers.
- Add a stress mode that replays synchronously on Thread_Gpu for bisecting.
- Tracy: Thread_Gpu per-draw time before and after; fresh-frame FPS A/B.

### Proposal (small): 7. Fix millisecond-granularity waits on Windows (GPU blocked-queue poll, Flip(0), guest timed waits)

**What.** Fix Common::CondVar::WaitFor on the Windows CRITICAL_SECTION path (threads.cpp:380-395), which converts micros to (micros < 1000 ? 1 : micros / 1000) ms:
- For micros == 0, return without sleeping.
- For waits under 1 ms, use a short spin/yield loop combined with SleepConditionVariableCS(0) retries, or a high-resolution waitable timer, re-checking the predicate.
- Stop truncating when rounding (micros / 1000 truncates).

In GuestGpu::ThreadRun, the all-queues-blocked case uses WaitFor(100) (graphicsRun.cpp:507). Replace it with an exponential backoff of about 10, 20, 50, 100, 200 µs using SleepMicro, and signal m_work_available from priority-thread callbacks and SendCommand.

**Why.** - The GPU thread re-evaluates WAIT_REG_MEM, CE/DE and occlusion-predicate suspensions (graphicsRun.cpp:299-372, 846-861) only after this timeout. Guest CPU label writes do not signal it. On Windows the intended 100 µs becomes 1 ms, since SDL3 raises the timer resolution to 1 ms (3rdparty/SDL3/src/timer/SDL_timer.c:578-587).
- FlipQueue::Flip(0) (videoOut.cpp:1113) is meant as a non-blocking poll but sleeps about 1 ms on Windows every vblank with no flip pending.
- Guest sema, event-flag and equeue waits shorter than 1 ms (semaphore.cpp:233-235, eventQueue.cpp:191) also round up to 1 ms.

**Impact.** Removes up to about 0.9 ms per all-queues-blocked episode and per sub-millisecond guest timed wait. The frame-time effect depends on how often these happen, so count WaitFor timeouts that are followed by progress. Likely small on average but helps frame-time consistency.

**Risks.** - More CPU wake-ups on an idle Thread_Gpu; bound the backoff.
- Guest waits must keep their 10 ms signal-polling behavior (poll_callback).
- Busy-waiting in guest waits would reintroduce the core-stealing problem from proposal 2, so prefer a yield-based loop.

**Validation.** - Add counters for blocked-queue waits and the time from WaitFor entry to a successful Process.
- Microbenchmark CondVar::WaitFor(0/100/500 µs) on Windows.
- Measure vblank jitter as in proposal 4.
- Run the sync_on_address and semaphore tests.

### Proposal (small): 8. Replace pure spin locks on CPU-fault paths with spin-then-park

**What.** - TrackingSpinLock (regionManager.h:29-50) and PageManager's SpinGuard (pageManager.cpp:61-73) spin on test_and_set with only atomic_signal_fence: no PAUSE, no yield, no park.
- Make them bounded-spin locks: _mm_pause for about 1-4k iterations, then SwitchToThread or sched_yield, then park with WaitOnAddress or a futex, with unlock waking one waiter. Keep the owner-recursion checks.
- For PageManager, avoid holding the region spin lock across the VirtualProtect/mprotect syscall (pageManager.cpp:203 onward, Protect at 190-195): collect ranges under the lock, then protect after dropping it, using a per-region sequence number to keep it correct.

**Why.** - Every guest CPU write fault on a GPU-tracked page takes the global TextureCache::m_lock (textureCache.cpp:1727) and a region lock in MemoryTracker::InvalidateRegion (memoryTracker.h:58-77), all on the faulting guest thread.
- Thread_Gpu holds m_lock across image lookup (textureCache.cpp:1319) and the full GC pass after every completed submission (2101, called from graphicsRun.cpp:610-645). It can even hold it while waiting on the GPU for a staging wrap (comment at textureCache.cpp:1957).
- MemoryTracker::ForEachUploadRange holds region locks across the upload memcpy when is_written (memoryTracker.h:111-127).
- Guest threads at HIGHEST priority then burn whole cores and SMT siblings spinning while the lock holder, or the priority thread it waits for, is starved.

**Impact.** Frees cores and SMT resources during Thread_Gpu-heavy phases and lowers priority-inversion risk. It is a stability and variance improvement with little average cost. Gain grows with the number of faulting guest threads.

**Risks.** - Keep acquire/release ordering identical.
- A parked waiter must not be missed on unlock (use a waiter count).
- Moving VirtualProtect out of the critical section needs care so a concurrent watch/unwatch cannot leave the wrong protection. If in doubt, keep that part as is.

**Validation.** - Run memory_tracker, performance_memory and the pending-writeback fault tests (Windows fault tests with gates).
- ETW: CPU time of guest threads inside TrackingSpinLock::lock before and after.
- A synthetic contention test with N threads faulting while one thread holds m_lock for 5 ms.

### Proposal (small): 9. Trim per-draw scheduler overhead on Thread_Gpu

**What.** - Before calling m_master.Refresh() in CommandScheduler::PopPendingOperations (commandScheduler.cpp:146-147), check that the pending queue is non-empty and that its front tick is above KnownGpuTick. Also let the priority thread publish the known tick after every wait.
- Without proposal 3, adopt a correctness-safe version of #506's release-mem flush batching: skip the Flush for data_sel=1 labels with no interrupt, because the label is already written at parse time.
- Move ShaderPrecompile journal appends (file IO through std::ofstream, pipelineCache.cpp:365) to a background writer.

**Why.** - PopPendingOperations runs at the start of every draw and dispatch (renderDraw.cpp:1195, 1306; renderCompute.cpp:202, 408). It unconditionally calls vkGetSemaphoreCounterValue, which is an ioctl on RADV and a driver call on Windows.
- RELEASE_MEM data_sel=1 flushes after every label (pm4Handlers.cpp:2321) even though WriteAtEndOfPipe memcpys the value at parse time (graphicsRun.cpp:1170-1175) and records no GPU work for it (sync.cpp:75-99).

**Impact.** Small: roughly 0.5-2 µs per draw, plus one vkQueueSubmit per label if proposal 3 is not done. Essentially free to implement.

**Risks.** - Deferred callbacks could run later if KnownGpuTick is stale. The priority thread refreshing the tick after its waits covers this.
- Removing label flushes delays the start of GPU work and interrupt delivery for queued priority ops; keep the flush whenever an interrupt is queued.

**Validation.** Tracy per-draw zone time, CTest predicate and EOP tests, and a check that interrupts and flips still fire on time (vblank and flip counters).

### Proposal (very large): 10. Evaluated, not recommended now: PM4 parse front-end / Vulkan back-end split

**What.** A front-end thread would decode PM4 and snapshot HW::Context, UserConfig and Shader state per draw. A back-end thread would do resource resolution and recording. To be correct, every PM4 handler with a guest-memory side effect must execute in back-end order, and any front-end read of memory the back-end may write must wait for the back-end to catch up.

**Why.** PM4 handlers touch guest memory synchronously in many places:
- EOP/RELEASE_MEM labels are memcpy'd at parse time (graphicsRun.cpp:1170-1175, 1543).
- WRITE_DATA memcpys (graphicsRun.cpp:396-426).
- WAIT_REG_MEM reads memory (graphicsRun.cpp:360-372).
- Draw/dispatch indirect arguments are memcpy'd from guest memory (graphicsRun.cpp:917-935, 1116-1121).
- Predicates read, and sometimes download, memory (graphicsRun.cpp:815-880).
- DMA_DATA goes through the buffer cache (graphicsRun.cpp:430-487).
- 8-bit index expansion reads guest indices during draw preparation (renderDraw.cpp:1263-1271).
- SRT materialization reads guest memory per draw (pipelineCache.cpp:285-292).

The expensive per-draw work (shader lookup, caches, descriptors) would land in the back-end anyway. The front-end would stall at every label, which Agc emits constantly.

**Impact.** Estimated at most 5-10% of Thread_Gpu, because PM4 decode is the cheap part, for a very large correctness-sensitive rewrite. The API-boundary split (proposals 3 and 6) gets most of the benefit with far less risk.

**Risks.** Reordering guest-visible writes relative to draw-time uploads gives wrong data: WRITE_DATA landing before an earlier draw's upload. Labels published before the draws they fence break guest synchronization. Drains would cost more because the front-end must also drain the back-end.

**Validation.** If ever attempted, run the PM4 test harness in lockstep mode (queue depth 1) against the current single-thread path and compare frame dumps bit-exactly.

### Proposal (very large): 11. Evaluated, defer: second VkQueue for guest async compute, secondary command buffers, CPU worker pool for detile and uploads

**What.** - **Second queue:** map the 56 guest compute queues (graphicsRun.h:47-51) to an AMD async-compute VkQueue with its own timeline semaphore and CommandScheduler, with cross-queue semaphore waits at WAIT_REG_MEM/label boundaries.
- **Secondary command buffers:** record render passes in parallel.
- **Worker pool:** a CPU pool for large staging memcpys, such as BufferCache::UploadCopies (bufferCache.cpp ~430-460) and priority-thread readback publication (bufferCache.cpp:176-183, textureCache.cpp:1953-1964).

**Why.** - **Async compute:** the host GPU is only 40% busy, so extra GPU overlap buys nothing. The CPU side of all queues would still run on Thread_Gpu, because BufferCache and TextureCache are GPU-thread-serialized (bufferCache.h:66-67; SlotVector/std::map without locks). Parallel recording hits the same wall: per-draw resolution mutates shared cache state (ResetBindings, image->binding in descriptors.cpp:744-751, Image::Transit layout tracking).
- **Detile:** already runs on the GPU (TileManager::Detile, textureCache.cpp:1087, 1120). What remains on the CPU is plain memcpy into staging or guest backing.

**Impact.** Near zero today. Revisit async compute after the drains are fixed: per-queue timelines could turn some full drains into waits on the compute queue only, but that belongs to the drain investigation. The memcpy pool helps only if profiles show copies of several MB per frame.

**Risks.** - **Async compute:** queue-family ownership transfers or CONCURRENT sharing for every resource; cross-queue hazards the cache layer does not model today; and much harder debugging.
- **Memcpy pool:** split copies must finish before the scheduler marks regions clean, and region spin locks are held across upload memcpys (memoryTracker.h:111-127).

**Validation.** Before any work, measure with RGP/RGA whether host GPU queue occupancy becomes the limit after proposals 1-6, and use Tracy to measure the memcpy share of Thread_Gpu time and of the priority thread.

### Proposal (medium): 12. Remove guest-side global serialization points: libUlt registry lock, memory-op lock held across the GPU unmap drain

**What.** (a) libUlt: UltMutexLock/Unlock and queue ops take the global g_ult_mutex and copy a shared_ptr on every call (libUlt.cpp:90-91, 103-114, 534-566). Store the state pointer inside the guest ULT object, as the pthread mutex does with atomic_ref (pthread.cpp:1700), or shard the map and use raw pointers kept alive by the create/destroy lifetime.

(b) Memory: munmap, mprotect-style paths and FreeGuestMemory hold the global recursive g_memory_operation_mutex (memory.cpp:830, 3755-3764) while UnmapGpuRange calls RenderContext::UnmapMemory. That calls SendCommandSync with Finish plus WaitPriorityOperations, a full drain on Thread_Gpu (renderContext.cpp:98-133). Make it two-phase:
- Mark the VA range as unmapping under the lock and drop the lock.
- Perform the GPU invalidate/unmap.
- Retake the lock to release the VA.

**Why.** - ULT-heavy engines would serialize all worker threads on one mutex and bounce one cache line.
- While one guest thread's unmap waits for a GPU drain, every other guest thread's mmap/munmap/mprotect/query and direct-memory call is blocked behind the same recursive mutex.

**Impact.** Conditional: it matters only if a trace shows libUlt calls or GPU-mapped unmaps during gameplay. The ported #483 already skips non-GPU ranges. When it applies, it can remove multi-millisecond stalls of unrelated guest threads.

**Risks.** - ULT object lifetime (destroy racing a lock) needs a defined rule.
- The two-phase unmap must stop the VA being handed out again (FindGuestFreeRange) and stop GPU faults on the range between phases.
- This overlaps with the drain investigation.

**Validation.** - Count libUlt calls per second and the time spent in g_memory_operation_mutex, using Tracy lock zones.
- Run kernel memory tests and the existing unmap timing harness (from the #483 port notes).

### Facts

- Guest x86-64 code runs natively: PthreadCreate calls host pthread_create(RunThread) (src/kernel/pthread.cpp:3286), and RunThread jumps to the guest entry with RunOnGuestStack (src/kernel/pthread.cpp:3218). Guest pthreads map 1:1 to host threads (winpthreads on Windows).
- Guest CPU affinity is stored but never applied to host threads (PthreadAttrSetaffinity, src/kernel/pthread.cpp:2075-2084; PthreadSetaffinity at 3379). The host scheduler places guest threads freely.
- On Windows, guest priority ≤478 maps to host +2 (THREAD_PRIORITY_HIGHEST) and ≥733 to -2 (src/kernel/pthread.cpp:2160-2170, 3499-3505). Emulator threads (Thread_Gpu graphicsRun.cpp:87, priority thread commandScheduler.cpp:100-104, present thread videoOut.cpp:251, precompile pipelineCache.cpp:540) are default-priority std::jthreads.
- Main thread runs the SDL event loop ('Thread_Window', window.cpp:709-729 via emulator.cpp:180). The guest main thread is a separate Common::Thread ('Thread_Main', emulator.cpp:173, runtimeLinker.cpp:1432).
- Thread_Gpu (GuestGpu::ThreadRun, graphicsRun.cpp:469-560) is the only consumer for queue 0 (graphics) and 56 compute queues (graphicsRun.h:47-51). It also runs SendCommand/SendCommandSync closures from guest threads, checked between every PM4 packet (graphicsRun.cpp:698-700).
- GuestGpu::Done (sceAgcSuspendPoint, agc.cpp:1551-1557) calls WaitForIdle while holding m_submission_mutex (graphicsRun.cpp:199-207, 461-466). The guest render thread blocks until Thread_Gpu has consumed all queued work. A queued alternative (commit 90bf3c92) was reverted the same day (ec78afb9) with no stated reason.
- Every guest submission slice ends with cp.BufferFlush(), which is vkQueueSubmit (graphicsRun.cpp:612, 638). RELEASE_MEM labels (data_sel=1) and interrupts also flush (pm4Handlers.cpp:2282, 2321, 2339). Submission is synchronous on Thread_Gpu under graphics.queue_mutex (commandScheduler.cpp:333-379).
- EOP/RELEASE_MEM label values are memcpy'd into guest memory at PM4 parse time, before GPU execution (graphicsRun.cpp:1170-1175; Sync::WriteAtEndOfPipe32 records only debug info, sync.cpp:75-99).
- There is a single Vulkan queue (queueCount=1, vulkanWindow.cpp:500-501, getQueue at 1076) and one command buffer being recorded at a time (CommandScheduler::m_command). The presenter has a second CommandScheduler with its own priority thread (swapchain.cpp:317, 354).
- RenderContext::m_mutex is held for the whole of each draw/dispatch (renderDraw.cpp:1203, 1314; renderCompute.cpp:221, 412; graphicsRun.cpp:1398). This covers shader compile (RefreshShaders reaching GetGraphicsPrograms), pipeline creation (pipelineCache.cpp:940) and resource preparation that may submit and wait (renderDraw.cpp:1098-1101).
- The present path takes the same RenderContext mutex (Presenter::PrepareFrame swapchain.cpp:725, PrepareBlankFrame 748, Present 798). The VideoOut present thread does VblankBegin, then Flip/Present, then VblankEnd (videoOut.cpp:833-869), so vblank events are delayed whenever Thread_Gpu holds the mutex.
- Common::CondVar::WaitFor on Windows+Clang uses SleepConditionVariableCS with (micros < 1000 ? 1 : micros/1000) ms (threads.cpp:385-388). The GPU thread's blocked-queue poll WaitFor(100) (graphicsRun.cpp:507) and FlipQueue::Flip(0) (videoOut.cpp:1113) become roughly 1 ms waits. SDL3 sets timeBeginPeriod(1) by default (3rdparty/SDL3/src/timer/SDL_timer.c:578-587).
- Guest KernelUsleep/Nanosleep (pthread.cpp:3844-3887) go through SleepMicroWithSignalPoll into Common::Thread::SleepMicro, which busy-spins (QPC+YieldProcessor on Windows, clock_gettime on Linux) for waits of 50 µs or less (threads.cpp:30-60, 135-160).
- Guest pthread mutex: a per-object std::mutex plus condition_variable, with no spin phase. The fast path has no global lock; lazy static init takes PthreadStaticObjects::m_mutex only once (pthread.cpp:1256-1320, 1410-1440).
- TextureCache::m_lock is a TrackingSpinLock that spins without pause or yield (regionManager.h:29-50). Every CPU write fault takes it (TextureCache::InvalidateMemory, textureCache.cpp:1727). Thread_Gpu holds it for lookups (1319), for the GC pass after each completed submission (2101), and while waiting on staging wraps (comment at 1957).
- PageManager's SpinGuard spins without pause and is held across VirtualProtect/mprotect in UpdateRegionWatchers (pageManager.cpp:61-73, 190-205).
- CPU write faults on pages that are not GPU-dirty are handled on the faulting guest thread (MemoryTracker::InvalidateRegion, memoryTracker.h:58-77). Only GPU-dirty pages route to Thread_Gpu with SendCommandSync (BufferCache::ReadMemory, bufferCache.cpp:247-275).
- Readback publication already runs on the CommandScheduler priority thread (DeferPriorityOperation in bufferCache.cpp:176-183 and textureCache.cpp:1953-1964). Texture detile/retile runs on the GPU through TileManager compute (textureCache.cpp:1087, 1120, 1763).
- Global g_memory_operation_mutex (recursive, memory.cpp:830) guards all guest memory ops. UnmapGpuRange (memory.cpp:92-97) reaches RenderContext::UnmapMemory, which does SendCommandSync(Finish+WaitPriorityOperations) under it (renderContext.cpp:98-133).
- PipelineCache: the precompile replay is a single jthread holding m_mutex through TranslateProgram (pipelineCache.cpp:540, 555-619). GetGraphicsPrograms/GetComputeProgram block on WaitForPrecompile (627, 692, 768). Runtime compiles and vkCreateGraphicsPipelines run synchronously on Thread_Gpu (356, 940, 972). Only shader permutations are journaled, not pipelines (shaderPrecompile.h:18-29).
- ProgramCache uses a shared member lookup_key (pipelineCache.cpp:408), so ProgramCache::Get is not reentrant without changes. The recompiler's only mutable statics are atomic log-once flags, so TranslateProgram looks reentrant.
- No VK_EXT_graphics_pipeline_library or shader-object path exists (grep over src finds none). Descriptors use push descriptors when available, otherwise vkUpdateDescriptorSets (descriptors.cpp:1117-1131).
- PopPendingOperations calls m_master.Refresh(), which is vkGetSemaphoreCounterValue, unconditionally at every draw and dispatch (commandScheduler.cpp:146-147; renderDraw.cpp:1195, 1306; renderCompute.cpp:202, 408).
- libUlt mutex lock/unlock and queue ops take the global g_ult_mutex and copy a shared_ptr on every call (libUlt.cpp:90-91, 103-114, 534-566).
- PrepareCpuFlip does Flush and then CompleteFlip immediately (graphicsRun.cpp:1553-1567). CPU flips therefore rely on VkQueue submission order between Thread_Gpu and the present thread's blit submit; any submit-thread design must keep a single FIFO.
- FFmpeg decoder contexts use the default thread_count of 1 (videoDec2Decoder.cpp:104-112, avPlayer.cpp:1114-1119), so FMV decode is single-threaded on the calling guest thread.
- Default logging is Silent (emulatorConfig.h:66), so LOGF and PRINT_NAME cost about one branch. In File mode, spdlog's _mt sink serializes all threads on its internal mutex (log.cpp:43-50).
- No open upstream PR addresses the threading architecture. #506 batches RELEASE_MEM flushes (an indication that per-label vkQueueSubmit cost matters). #35 proposes an async pipeline builder but is low quality: it adds test_shader.spv and duplicate headers.



## Second pass (September 25, 2026): the areas that did not finish

These notes were written by reading the code at revision 964ebc6, whose `src/` is identical to
791ac3a, and by checking them against the Thread_Gpu stack sample from the September 24 Astro Bot
run. As above, file:line references are leads to verify, not guarantees.

### Correction: the "12% shader translation" is per-draw resource materialization

The two leaf functions behind the 12% are `SrtWalker::EvaluateWide` (8.1%) and `Value::Resolve`
(3.5%). They are not compile-time work. The profiled run replayed the shader journal at startup
("replayed 307 permutations; skipped 0") and compiled nothing afterwards: `ProgramCache::Get`
prints an unconditional `Shaders: VS … | PS …` line for every new permutation
(pipelineCache.cpp:379-391), and the run's stdout has none. The hot caller is therefore the cache
hit path, where `ProgramCache::Get` calls `MaterializeResources` on every draw and dispatch for
every stage (pipelineCache.cpp:297-300).

`MaterializeResources` (ResourceMaterialization.cpp:932-1058) builds two `SrtWalker`s, a clean one
and a normal one. It runs `FindActiveSources`, re-evaluates every flat SRT slot
(`RefreshFlatBuffer`, SrtWalker.cpp:1041-1058), and then walks every buffer, image and sampler
descriptor dword through the IR. The walker is a recursive IR interpreter. Every argument goes
through `Value::Resolve()` (an identity-chain walk, Value.cpp:68-70), a dense-memo generation
check (SrtWalker.cpp:515-527) and an opcode switch (SrtWalker.cpp:648-987). `ReadFirstLane`
builds two nested walkers per evaluation (SrtWalker.cpp:668-674). Clean reads go through
`ReadShaderGuestMemory` → `TryReadGpuCleanBacking` (memory.cpp:881-890). Each such read does a
`std::map` range lookup (`HasGpuDirtyBytes`) plus `TextureCache::IsRegionGpuModified`, which
takes the texture-cache spin lock and walks the image page table (textureCache.cpp:1995-2007).
The results are then compared field by field against every permutation in a linear search
(pipelineCache.cpp:301-309).

Ideas, in order of expected payoff:
- **Memoize materialization per source entry.** Keep the last user-data span and the last flat
  SRT buffer for each `SourceEntry`. When a plan has no dynamic reads (`dynamic_reads` empty) and
  no clean-slot or indirect-image dependencies, the snapshot and specialization are a pure
  function of the user data, the shader base and the flat slot values. After `RefreshFlatBuffer`,
  a memcmp against the previous inputs can skip every descriptor evaluation and the permutation
  search. Risk: an undeclared memory dependency returns a stale descriptor. Such plans must be
  excluded conservatively, and a debug mode should re-evaluate and compare.
- **Compile the plan to a flat evaluation program** at `ExtractResourcePlan` time: topologically
  ordered, identities pre-resolved, operands as slot indices. That removes the recursion,
  `Resolve()` chains and memo checks. Estimated 3-10x on the walker's own cost.
- **Skip the texture-cache lock for clean reads** by keeping a GPU-dirty summary that can be read
  without the lock, or by reading the tracker's GPU bits first.
- **Put the last-hit permutation at the front** of `SourceEntry::permutations`, as the first
  survey already proposed.

Measure with inline-aware symbolization (see "Profiling notes" below) before choosing.

**Status (September 25, branch `perf/srt-materialization`).** The first three ideas are
implemented; the texture-cache lock itself is untouched, but strict reads now come from 64-byte
blocks. A replay of the 308-record Astro Bot shader journal (279 distinct plans) gives the plan
shapes: 157 value nodes, 36 flat SRT slots and 55 descriptor dwords per plan on average; 121
plans have branch conditions, 37 have selects and none has `ReadFirstLane` or indirect images.
276 of the 279 plans qualify for the memo. The compiled evaluator matches the walker exactly on
every plan and on four synthetic memory patterns, and is about 2.9 times faster per refresh.
Separately, 305 of the 308 shaders have no declared hash, so `GetShaderParams` computes XXH3 over
about 4.4 KB of code for every stage of every draw. That is the ~5% `GetShaderParams` share above,
and it becomes the larger per-draw cost once materialization is cheaper.

### First-use shader compile cost (September 25)

A new shader costs our recompile plus the AMD driver's pipeline compile, and both happen on first
encounter. The recompile also repeats at boot for every journal record. Measured on the
7800X3D with a copy of the 308-record Astro Bot journal:
- **Driver:** `vkCreateComputePipelines` takes 1.27 s for `900aba8df9448d3d` (169k SPIR-V words)
  and 0.85 s for `01d6f21611218e74` (133k). A cache hit costs about 0.1 ms, so these only hurt
  when the driver cache is cold.
- **SPIR-V composition** of the big shaders: `OpBitcast` about 19% of words, `OpSelect` 15-19%
  (before the select-chain collapse), `OpPhi` about 10%. 28% of IR bit casts repeat an earlier
  one with the same source, and `900aba` has about 3,200 duplicate pure instructions, which a
  CSE pass could remove. Every buffer dword load is two nested selection constructs: an EXEC
  branch, then a software bounds check (`OpArrayLength`, `OpULessThan`, branch, phi). In
  `900aba`, 657 of 799 `OpULessThan` are these checks. The device enables `robustBufferAccess2`
  when supported, and that already returns zero for out-of-bounds loads.
- **Recompiler, by pass:** in the base build, `CFG::Structurize` took 1318 of 3471 ms and 63-77%
  of each big shader. Almost all of it was `ComputePostDominators`, rerun after each of 27-62
  merge splits and iterated in forward block order over block-sized sets. Reverse order fixed
  it (commit 3af11427, byte-identical). The rest is now `TranslateProgram` about 616 ms,
  `RewriteToSsa` 427, `EmitProgram` 416, constant propagation 186, `BuildSrtPlan` 107 and
  `RemoveIdentities` 106.
- **Use lists:** `Inst::AddUse` scans the used value's whole use list for a duplicate before
  every append (always on; `KYTY_FINAL` is not defined), and `ReplaceUsesWith` erases one use at
  a time. Removing both kept the SPIR-V identical and made `RewriteToSsa` about 16% faster, but
  that run overlapped emulator boots and other passes moved too. `RemoveIdentities` did not
  change. Re-measure on an idle machine before committing it.
- **SSA construction** seals every block only after the whole program is visited, so each read
  without a local definition creates an incomplete phi. Sealing earlier would change phi order
  and therefore the SPIR-V bytes. It is only worth doing with a GPU and correctness A/B.
- **What the driver pays for** (`--pipeline-compile-time`, compute shaders, 4-5 interleaved
  rounds). Control flow dominates:
  - Merging block-local and dominated duplicate pure instructions cut `900aba` by 6% in words
    but its compile by only 1.2% (`01d6f2` 4% and 1.8%). Dropped. It also has to rewrite branch
    conditions, descriptor sources and SRT reads, which hold IR values outside the use lists; a
    first version left a dangling branch condition.
  - Replacing the bounds branch with an index clamp cut 7% of words and 22% of compile time
    (1397 to 1096 ms; `01d6f2` 964 to 798 ms).
  - Also dropping the EXEC branch around loads gave a further 13% on `900aba` but was 10% worse
    on `01d6f2`, and it changes what inactive lanes read. Not pursued.
  - `min(index, length)` alone is not exact, because a range that is not a multiple of 4 is
    rounded up by `robustBufferAccess2`, so dword `length` can be partly in range.
  - 6820afbd used `v = load(in ? index : 0); in ? v : 0`. The cold A/B improved (graphics
    pipelines -9%), but warm GPU zones regressed: `ef31694e` +17.8% µs/Mpx, `5a10a907` +6.2%,
    `969c5ed1` +4.3%. `ef31694e`'s only bounds-checked loads are 94 scalar constant-buffer reads.
    The selected address blocks AMD's constant-offset folding (`offset:imm`) and neighbour
    merging.
  - b3ffaa5f loads at the untouched index (`in ? load(index) : 0`, safe because
    `robustBufferAccess` is always on) and keeps the branch for LDS and scratch. Warm: every
    zoned shader was 3.8-16% cheaper and GPU busy was unchanged. Cold: graphics 15.2 to 13.0 s,
    compute 5.87 to 5.40 s. Adopted.
  - A "yellow sand" run during this A/B came from the perf session's mesh pipeline libraries:
    a baseline run showed it too. Compare screenshots across all runs before blaming a change.
- **Stores** keep their bounds branches: 181 storage stores in `01d6f2`, and 249-307 LDS stores in
  the big mesh shaders. A branch-free storage store needs an index that is guaranteed out of
  range (for example 0x3FFFFFFF) and a discard guarantee, which only `robustBufferAccess2`
  gives, so the device feature would have to reach the recompiler. LDS stores could go to one
  extra "discard" dword past the guest's LDS size when the host limit allows. Both change GPU
  code, so both need the same warm-zone gate.

### Texture-cache lookups

Per draw: `PrepareDrawRenderState` (renderDraw.cpp:898-935) resolves every MRT slot and the depth
target through `FindImage` (textureCache.cpp:1305-1401). Descriptor preparation resolves every
sampled and storage image the same way. Each lookup:
- takes `m_lock` (a pure spin lock) and runs `FindImagesInRegion`, which visits 1 MiB image-page
  buckets (textureCache.h:97) and deduplicates by query epoch, then `SameBacking` for each
  candidate;
- on a miss, runs `ResolveOverlap` over the candidates, possibly with `FreeImage`,
  `ExpandImage` or copies;
- under VRAM pressure, may run `CollectGarbage(true)` at most once per GC tick
  (textureCache.cpp:1358-1368);
- always calls `MaterializeDccClear` (textureCache.cpp:1387) after the lock is released. For native
  DCC surfaces whose metadata range is GPU-dirty, that is a synchronous `ReadMemory` full drain
  (textureCache.cpp:1210-1214). The September 24 sample showed exactly this chain:
  `ResolveRenderColorTarget` → `FindImage` → `BufferCache::ReadMemory` → `CommandScheduler::Wait`.
  When `MaterializeDccClear` consumes a clear, it writes the metadata with `FillBuffer(UINT32_MAX)`
  on the CPU (textureCache.cpp:1242). If that range is GPU-dirty again, the fill goes through the
  GPU path (bufferCache.cpp:524-533) and re-arms the drain for the next lookup.
- `FindTexture` re-runs `RefreshImage` (textureCache.cpp:1463). For images marked "maybe CPU
  dirty", this hashes guest edges (`HashGuestEdges`, textureCache.cpp:1255-1262) on every
  binding.

Open questions for the instrumentation: how many `FindImage` calls per frame, how many hit the
DCC drain, and how many run `HashGuestEdges`.

### DMA path

`CommandProcessor::DmaData` (graphicsRun.cpp:397-450) has no wait of its own. Its drains can only
come from these inner calls:
1. **CPU fast paths.** `CopyBuffer` memcpys when neither range is GPU-dirty and no image starts at
   the source (bufferCache.cpp:549-554). `FillBuffer` does `std::fill` when the destination is
   not GPU-dirty (bufferCache.cpp:524-528). A write to a clean, write-watched page faults on
   Thread_Gpu. `HandleFault` then marks the page CPU-dirty without a drain
   (memoryTracker.h:62-77), and `TextureCache::InvalidateMemory` drains only if an image
   writeback is pending on that page (textureCache.cpp:1728-1742).
2. **GPU paths.** `ObtainBuffer` → `SynchronizeBuffer` → `UploadCopies` → `m_staging_buffer.Map`
   (bufferCache.cpp:427). A 512 MiB staging-ring wrap waits for the previous lap's ticks
   (streamBuffer.cpp:313-333). That is a full drain only when the lap happened inside the open
   command buffer.
3. **Buffer creation.** `FindBuffer` → `CreateBuffer` → `Register` → `WriteDataBuffer`
   (bufferCache.cpp:79-80) copies BDA page-table entries through the same staging ring.
4. **Texel sources.** `ObtainBuffer(src, …, is_texel_buffer=true)` → `SynchronizeBufferFromImage`
   (textureCache.cpp:1837-1901) records an image download into the buffer. That records work but
   does not wait.

The September 24 stack attributed about 13% of Thread_Gpu samples to `DmaData` → `Wait`, with two
mislabeled frames between them. Only (2) or (3), a staging-ring wait, fits a direct call chain
with no exception frames. An exception-driven drain would show `KiUserExceptionDispatcher` and
`HostException::ExceptionFilter` frames, as the third stack did. Staging usage between two
submits is the thing to count.

### Fault handling

Path: Windows VEH `ExceptionFilter` (hostException.cpp:85-140) → `KytyExceptionHandler`
(runtimeLinker.cpp:782-803) → `HandleGpuFault` (memory.cpp:929-931) → `RenderContext::HandleFault`
(renderContext.cpp:58-72). The handler costs, in order:
- a kernel exception dispatch of roughly 2-5 µs, unavoidable with page protection;
- a `shared_mutex` read lock for `IsMapped`;
- the tracker region `TrackingSpinLock`;
- for writes, `TextureCache::m_lock`;
- the page-protection syscall (VirtualProtect) under the PageManager region spin lock
  (pageManager.cpp:200-249).

Reads of GPU-dirty pages always go to Thread_Gpu (`SendCommandSync`, graphicsRun.cpp:144-156),
which serves them only between PM4 packets or when idle. While it is inside a drain, other
faulting threads queue behind it. Only reads of GPU-dirty pages drain; writes to clean tracked
pages are resolved on the faulting thread.

The fault is reported for one byte, so the 512 KiB window in `ReadMemory` (bufferCache.cpp:259-264)
is what amortizes consecutive reads. Every page in the window becomes readable after one drain.

### Command-processor waits

- **WAIT_REG_MEM** (graphicsRun.cpp:337-348) reads the address on the CPU. If the page is
  GPU-dirty, that read is a Thread_Gpu self-fault with a full drain. If the value does not match,
  the queue is suspended. When every queue is blocked, `ThreadRun` waits
  `m_work_available.WaitFor(…, 100)` (graphicsRun.cpp:505-513), which is 1 ms on Windows because
  `CondVar::WaitFor` rounds sub-millisecond waits up to 1 ms (threads.cpp:387). Guest label writes
  from the CPU do not signal `m_work_available`, so a blocked queue is re-checked only by that
  poll, by a new submission, or by a command.
- **CE/DE counters** (graphicsRun.cpp:286-297) suspend the same way.
- **SET_PREDICATION** with `wait_op` goes through `SynchronizePredicate`; see the first survey.
- **COND_EXEC-style reads** (pm4Handlers.cpp:1479) and indirect draw/dispatch arguments
  (graphicsRun.cpp:928-1125) dereference guest memory on the CPU and self-fault if it is
  GPU-dirty.

What to count: blocked-queue polls per second and their total time, suspended packets by opcode,
and self-faults by opcode.

### Per-draw CPU cost (other than materialization)

For each draw (`DrawIndex`, renderDraw.cpp:1189-1298):
- `PopPendingOperations`, which always calls `vkGetSemaphoreCounterValue`
  (commandScheduler.cpp:211);
- the `RenderContext::m_mutex` lock;
- `RefreshShaders` → `GetGraphicsPrograms` (key building, materialization, permutation search);
- colour and depth `FindImage` lookups;
- `PrepareBindings` for every stage, then `PrepareGraphicsBindings` (buffer and texture
  resolution, BDA sync when needed);
- `AcquireVertexBuffers` and `ObtainBuffer` for indices; 8-bit indices are expanded on the CPU
  into a `std::vector` (renderDraw.cpp:1263-1273);
- `AcquireRenderTargets`;
- `GetGraphicsPipeline` (a ~720-byte key hashed byte by byte);
- `CommitBindings` (push descriptors), dynamic state, `BeginRendering`, `bindPipeline` and the
  draw itself.

`SetVulkanObjectNameF` and `LogDrawPhase` return early unless debug dumps are on, so they are
free.

The fault-buffer parser (faultManager.cpp:77-149) is dispatched once per completed submission that
used BDA. It covers `CACHING_NUMPAGES / 32` = 2M threads, scanning the whole 8 MiB bitmap each
time. That is GPU time, not CPU, and the GPU is not the bottleneck today.

### Host synchronization

- `MasterSemaphore::Wait` (masterSemaphore.cpp:38-55) checks the cached tick, refreshes, and then
  blocks in `vkWaitSemaphores`. The AMD driver implements that wait with a kernel event (the
  `NtWaitForSingleObject` leaves in the sample), so wake-up latency after the GPU signals is
  scheduler-bound, typically tens of µs.
- The priority thread (commandScheduler.cpp:268-295) is a default-priority thread that host-waits
  each op's tick, then runs it. Each drain needs two wake-ups: the waiter, and the priority thread
  for `WriteBacking`. The waiter then needs a third wake-up through `m_operation_available` in
  `WaitPriorityOperations`.
- `Submit` holds `queue_mutex` across `vkQueueSubmit` (commandScheduler.cpp:359-380).
  `vkQueuePresentKHR` holds the same mutex (swapchain.cpp:696-699), so a present in progress
  delays the submit inside every drain.
- `CondVar::WaitFor` on Windows uses `SleepConditionVariableCS` with millisecond granularity and
  rounds anything under 1 ms up to 1 ms (threads.cpp:379-396).

### Frame pacing

- The VideoOut `PresentThread` (videoOut.cpp:801-874) is both the vblank generator and the
  presenter. Each 60 Hz period it runs `VblankBegin`, then `Flip(0)`, then `VblankEnd`.
  `Flip(0)` with an empty queue calls `WaitFor(…, 0)`, which sleeps 1 ms on Windows
  (videoOut.cpp:1113, threads.cpp:387). Vblank-end events therefore arrive at least 1 ms after
  vblank-begin whenever no flip is ready.
- `Present` (swapchain.cpp:781-819) takes `RenderContext::m_mutex` to record and submit the blit
  (swapchain.cpp:798-804). Thread_Gpu holds that mutex for the whole of every draw and dispatch,
  including drains inside them. The present, and the `VblankEnd` event after it, can therefore
  slip by as long as one drain.
- The default present mode is Mailbox (emulatorConfig.h:49). `vkAcquireNextImageKHR` uses an
  infinite timeout, which is normally non-blocking with Mailbox.
- The window title reports game fps as fresh frames. At about 27 fps the game flips on roughly
  every other 60 Hz vblank, so a frame slightly over 33.3 ms costs a whole vblank. That is
  consistent with the observed jumps between about 20 and 40 fps.

### Upstream PR performance triage (refreshed September 25)

Upstream `main` has not moved since 5a705dd. These open PRs changed since the first snapshot and
touch performance:
- **#822** (AudioOut write cadence, 28 lines): paces blocking audio writes to one buffer period.
  This is an audio-quality fix. It adds sleeps to the guest audio thread, which is not on the
  frame's critical path. Neutral for fps. It can be ported for audio if wanted.
- **#789** (FNAF memory fixes): validates every buffer and sampler descriptor inside
  `MaterializeResources`, which runs per draw (see above). It adds a `ClampRangeSize` lookup per
  buffer per draw. Correctness work that would make the hot path slower. Do not port as is.
- **#720** (Astro Bot Intel compatibility): x86 instruction emulation for Intel CPUs, plus
  audio-pointer validation with a VirtualQuery per `AudioOutOutputs` call. It does nothing for the
  7800X3D. Skip.
- **#748** (async logging, VMA and Tracy frame profiling, macOS ARM): the Tracy frame markers could
  help measurement, but the PR is mostly platform work. Take the ideas, not the commits.
- **#719** (asynchronous logger): only matters with file logging, which should stay off for
  timing.
- **#715, #741, #727, #643** (carbonimax graphics series: native PerVertex replay, mesh-output
  pruning, wave32 NGG): large, and conflict heavily with this branch's renderer. #741 can reduce
  GPU mesh-shader work, but the GPU is not the bottleneck.
- **#811** (address-backed indirect image descriptors): changes resource tracking and
  materialization. Re-measure per-draw cost if it lands upstream.
- **#558** (Astro Bot RT and an "SRT plan bug that dropped its lighting"): the SRT fix may matter
  for correctness with the non-RT patch. Needs a separate look.

No open PR addresses the drains, the per-draw materialization cost, or the Thread_Gpu
bottleneck.

### Profiling notes

The September 24 stacks had implausible frames (for example `RangeSet::Subtract` calling
`CommandScheduler::Wait`) because ThinLTO inlining and identical-code folding merge functions, and
`SymFromAddr` returns only the outer symbol. The sampler should resolve inline frames
(`SymAddrIncludeInlineTrace`, `SymQueryInlineTrace`, `SymFromInlineContext`) and file:line
(`SymGetLineFromInlineContext`) before anyone attributes time at the call-site level.
