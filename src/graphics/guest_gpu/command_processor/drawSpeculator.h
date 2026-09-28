#ifndef GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_DRAW_SPECULATOR_H
#define GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_DRAW_SPECULATOR_H

#include "common/common.h"
#include "graphics/host_gpu/renderer/pipeline/drawSpeculation.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

namespace Libs::Graphics {

class CommandProcessor;
class RenderContext;

// Refreshes the shader resources of draws ahead of the GPU thread (Config::SpeculativeDrawsEnabled;
// KYTY_SPECULATE_DRAWS=0 or 1 overrides it). A
// worker thread replays the command stream's register writes from a copy of the GPU thread's
// register state, as the pipeline look-ahead does, and speculates each draw it meets
// (PipelineCache::SpeculateGraphicsPrograms). Both threads number draw packets from the copy's
// position; the GPU thread takes a draw's speculation when it is ready and never waits for one.
// A wrong speculation costs nothing but the worker's time: adoption checks every input again.
class DrawSpeculator {
public:
	struct Cursor {
		std::span<const uint32_t> commands;
		uint32_t                  offset_dw = 0;
	};

	DrawSpeculator(RenderContext& renderer, int interrupt_event_id);
	~DrawSpeculator();
	KYTY_CLASS_NO_COPY(DrawSpeculator);

	[[nodiscard]] static bool Enabled();

	// GPU thread: execution continues at `stack` with `processor`'s register state (a submission
	// starting or resuming, or a draw past a packet the walk stopped at).
	void Restart(const CommandProcessor& processor, std::span<const Cursor> stack);
	// GPU thread, before each draw packet's handler, in order: the draw's speculation, if ready.
	[[nodiscard]] SpeculatedDraw* Take(const uint32_t* packet);
	// GPU thread, after the handler (with Take's result).
	void Release(SpeculatedDraw* draw);
	// GPU thread, after Release: whether the walk has stopped behind the GPU thread and needs a
	// Restart at the next packet.
	[[nodiscard]] bool RestartDue() const;

private:
	enum SlotState : uint32_t { Free, Writing, Ready, Taken };
	struct Slot {
		std::atomic<uint32_t> state {Free};
		uint64_t              epoch  = 0;
		uint64_t              seq    = 0;
		const uint32_t*       packet = nullptr;
		SpeculatedDraw        draw;
	};
	struct Snapshot;
	static constexpr uint32_t RingSize = 32;
	static constexpr uint64_t NotStopped = UINT64_MAX;
	// Draws within this many of the GPU thread's next draw are not speculated (see Walk).
	static constexpr uint64_t SpeculationLead = 2;

	void Run();
	// Walks from the current snapshot until it stops, speculating the draws it meets.
	void Walk(CommandProcessor& shadow, std::vector<Cursor> stack, uint64_t epoch, uint64_t seq);
	// A free slot for draw `seq`, or null when a restart or shutdown interrupted the wait.
	Slot* AcquireSlot(uint64_t seq, uint64_t epoch);
	// Waits until the GPU thread has finished draw `seq`; false when interrupted.
	bool  WaitForGpu(uint64_t seq, uint64_t epoch);
	[[nodiscard]] bool Interrupted(uint64_t epoch) const;
	// GPU thread: whether `draw`'s reads still give what they gave (see Take).
	[[nodiscard]] bool ReadsCurrent(const SpeculatedDraw& draw);

	RenderContext& m_renderer;
	const int      m_interrupt_event_id;

	std::array<Slot, RingSize> m_ring;
	// GPU thread: the next draw's sequence number and the current epoch.
	uint64_t                   m_gpu_seq = 0;
	// GPU thread: the buffer cache's and texture cache's GPU-write generations at the last
	// Release, and the last draw at whose Release either had changed (see Take).
	uint64_t                   m_seen_buffer_writes = 0;
	uint64_t                   m_seen_image_writes  = 0;
	uint64_t                   m_last_write_seq     = 0;
	std::atomic<uint64_t>      m_gpu_next {0};
	std::atomic<uint64_t>      m_epoch {0};
	// The worker's walk ended before this draw (NotStopped while walking or idle at the end).
	std::atomic<uint64_t>      m_stopped_seq {NotStopped};
	bool                       m_mismatch = false;

	std::mutex                 m_mutex;
	std::condition_variable    m_wake;
	std::unique_ptr<Snapshot>  m_snapshot; // Pending restart, under m_mutex.
	std::unique_ptr<Snapshot>  m_spare;    // The worker's; swapped with m_snapshot to take it.
	bool                       m_restart = false;
	// Set with m_restart, read without the lock: the worker spins on it before sleeping.
	std::atomic<bool>          m_restart_pending {false};
	bool                       m_quit    = false;
	std::atomic<bool>          m_waiting {false};
	// While the worker waits: the GPU thread's draw count that wakes it (a waking per draw would
	// cost the GPU thread a kernel call each).
	std::atomic<uint64_t>      m_wake_at {0};
	std::atomic<bool>          m_quitting {false};
	// Bumped when the worker may stop waiting for a slot (a draw finished, a restart, shutdown).
	std::atomic<uint64_t>      m_signal {0};
	std::thread                m_thread;

	// Counters, printed with KYTY_DEBUG_SPEC_STATS=1.
	struct Stats {
		std::atomic<uint64_t> restarts {0};
		std::atomic<uint64_t> walked {0};
		std::atomic<uint64_t> speculated {0};
		std::atomic<uint64_t> behind {0};
		std::atomic<uint64_t> taken {0};
		std::atomic<uint64_t> missed {0};
		std::atomic<uint64_t> mismatched {0};
		std::atomic<uint64_t> deferred {0};
		std::atomic<uint64_t> table_unread {0};
		std::atomic<uint64_t> table_unknown {0};
		std::array<std::atomic<uint64_t>, 256> stops {};
	};
	Stats m_stats;
	void  PrintStats();
	void  Stop(uint64_t seq, uint64_t epoch, uint32_t opcode);
	void  Signal();
};

} // namespace Libs::Graphics

#endif // GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_DRAW_SPECULATOR_H
