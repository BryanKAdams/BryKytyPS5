#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDHOOKS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDHOOKS_H_

#include "common/common.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <stop_token>

namespace Libs::Graphics {

// Hooks the Vulkan functions the renderer records commands with (and descriptor updates, command
// buffer and pool calls, queue submits) in the default dispatcher. Install once the device
// dispatcher is initialized. Without either mode below the dispatcher is left alone.
// - Deferred recording (Config::RecordThreadEnabled(); KYTY_RECORD_THREAD=0 or 1 overrides it):
//   commands any thread records into a stream's routed command buffer are copied into that
//   CommandStream and recorded by its consumer (the render scheduler's submit thread) in the
//   same order, together with the submits. Other commands into that buffer run after what the
//   stream holds, recorded directly.
// - Otherwise KYTY_DEBUG_VK_TIME=1: the GPU thread's calls are timed; a line every 5 s says how
//   much of the thread they took.
void InstallCommandHooks();

// The GPU thread's current thread counts as the GPU thread for the timing.
void MarkCommandHookThread();

// Whether InstallCommandHooks installed the recording hooks.
[[nodiscard]] bool CommandRecordingDeferred();

// On a thread that routes a stream (the GPU thread): waits until that stream has recorded and
// submitted everything appended so far. Call before submitting to the queue directly, and before
// taking the queue lock.
void DrainGpuThreadCommands();

// An ordered stream of recorded Vulkan calls and other work, from the thread that routes it (and
// any thread recording into its routed buffer) to one consumer. At most four exist at a time.
class CommandStream {
public:
	CommandStream();
	~CommandStream();
	KYTY_CLASS_NO_COPY(CommandStream);

	// From now on, Vulkan commands recorded into `buffer` (and this thread's descriptor updates)
	// go into this stream; a null buffer ends the routing.
	void Route(VkCommandBuffer buffer);
	// Appends a call to run in order with the recorded commands.
	void Push(Common::UniqueFunction<void>&& call);
	// Wakes the consumer if it sleeps (after a submit, or with much work queued).
	void Wake();
	// Waits until the consumer has run everything appended so far.
	void Drain();
	// Consumer: runs appended work until `stop` is requested and nothing is left.
	void Consume(std::stop_token stop);

	struct Impl;
	struct Slot;

private:
	Slot* m_slot = nullptr;
	Impl* m_impl = nullptr;
};

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDHOOKS_H_ */
