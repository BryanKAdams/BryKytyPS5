#include "graphics/guest_gpu/command_processor/drawSpeculator.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/command_processor/pm4Dispatch.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "kernel/memory.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <immintrin.h>

namespace Libs::Graphics {

namespace {

bool EnvEnabled(const char* name) {
	const char* value = std::getenv(name);
	return value != nullptr && std::strcmp(value, "1") == 0;
}

// As ProcessPm4 counts draw packets.
bool IsDrawPacket(uint32_t opcode) {
	switch (opcode) {
		case Pm4::IT_DRAW_INDEX_2:
		case Pm4::IT_DRAW_INDEX_OFFSET_2:
		case Pm4::IT_DRAW_INDEX_AUTO:
		case Pm4::IT_DRAW_INDIRECT:
		case Pm4::IT_DRAW_INDEX_INDIRECT:
		case Pm4::IT_DRAW_INDIRECT_MULTI:
		case Pm4::IT_DRAW_INDEX_INDIRECT_MULTI:
		case Pm4::IT_DISPATCH_DRAW_PREAMBLE: return true;
		default: return false;
	}
}

// Packets that neither change register state nor decide which packets run next. Memory they
// write may change what a later draw reads; adoption finds that out. WAIT_ON_CE_COUNTER is left
// out: the constant engine may dump the register tables the draws after it load.
bool PassesRegisters(uint32_t opcode) {
	switch (opcode) {
		case Pm4::IT_SET_BASE:
		case Pm4::IT_INDEX_BUFFER_SIZE:
		case Pm4::IT_DISPATCH_DIRECT:
		case Pm4::IT_DISPATCH_INDIRECT:
		case Pm4::IT_INDEX_BASE:
		case Pm4::IT_CONTEXT_CONTROL:
		case Pm4::IT_INDEX_TYPE:
		case Pm4::IT_NUM_INSTANCES:
		case Pm4::IT_WRITE_DATA:
		case Pm4::IT_MEM_SEMAPHORE:
		case Pm4::IT_WAIT_REG_MEM:
		case Pm4::IT_WAIT_REG_MEM_64:
		case Pm4::IT_COPY_DATA:
		case Pm4::IT_CP_DMA:
		case Pm4::IT_PFP_SYNC_ME:
		case Pm4::IT_SURFACE_SYNC:
		case Pm4::IT_EVENT_WRITE:
		case Pm4::IT_EVENT_WRITE_EOP:
		case Pm4::IT_EVENT_WRITE_EOS:
		case Pm4::IT_RELEASE_MEM:
		case Pm4::IT_DMA_DATA:
		case Pm4::IT_ACQUIRE_MEM:
		case Pm4::IT_WRITE_CONST_RAM:
		case Pm4::IT_DUMP_CONST_RAM:
		case Pm4::IT_INCREMENT_CE_COUNTER:
		case Pm4::IT_INCREMENT_DE_COUNTER:
		case Pm4::IT_WAIT_ON_DE_COUNTER_DIFF:
		case Pm4::IT_GET_LOD_STATS: return true;
		default: return false;
	}
}

} // namespace

struct DrawSpeculator::Snapshot {
	HW::Context         ctx;
	HW::Context         saved_ctx;
	bool                context_state_pushed = false;
	HW::UserConfig      ucfg;
	HW::Shader          sh_ctx;
	HW::UserSgprType    user_data_marker = HW::UserSgprType::Unknown;
	std::vector<Cursor> stack;
	uint64_t            epoch = 0;
	uint64_t            seq   = 0;
};

DrawSpeculator::DrawSpeculator(RenderContext& renderer, int interrupt_event_id)
    : m_renderer(renderer), m_interrupt_event_id(interrupt_event_id),
      m_snapshot(std::make_unique<Snapshot>()), m_spare(std::make_unique<Snapshot>()) {}

DrawSpeculator::~DrawSpeculator() {
	{
		std::scoped_lock lock(m_mutex);
		m_quit = true;
	}
	m_quitting.store(true, std::memory_order_release);
	m_wake.notify_one();
	Signal();
	if (m_thread.joinable()) {
		m_thread.join();
	}
}

bool DrawSpeculator::Enabled() {
	static const int forced = [] {
		const char* value = std::getenv("KYTY_SPECULATE_DRAWS");
		return value == nullptr ? -1 : std::strcmp(value, "0") == 0 ? 0 : 1;
	}();
	return forced >= 0 ? forced != 0 : Config::SpeculativeDrawsEnabled();
}

void DrawSpeculator::Signal() {
	m_signal.fetch_add(1, std::memory_order_release);
	m_signal.notify_one();
}

bool DrawSpeculator::Interrupted(uint64_t epoch) const {
	return m_epoch.load(std::memory_order_acquire) != epoch ||
	       m_quitting.load(std::memory_order_acquire);
}

void DrawSpeculator::Restart(const CommandProcessor& processor, std::span<const Cursor> stack) {
	const auto epoch = m_epoch.load(std::memory_order_relaxed) + 1u;
	m_epoch.store(epoch, std::memory_order_release);
	m_stopped_seq.store(NotStopped, std::memory_order_relaxed);
	m_mismatch = false;
	{
		std::scoped_lock lock(m_mutex);
		auto&            snapshot     = *m_snapshot;
		snapshot.ctx                  = processor.m_ctx;
		snapshot.context_state_pushed = processor.m_context_state_pushed;
		// A context state pop restores it; without a push, the next push overwrites it.
		if (processor.m_context_state_pushed) {
			snapshot.saved_ctx = processor.m_saved_ctx;
		}
		snapshot.ucfg                 = processor.m_ucfg;
		snapshot.sh_ctx               = processor.m_sh_ctx;
		snapshot.user_data_marker     = processor.m_user_data_marker;
		snapshot.stack.assign(stack.begin(), stack.end());
		snapshot.epoch = epoch;
		snapshot.seq   = m_gpu_seq;
		m_restart      = true;
		m_restart_pending.store(true, std::memory_order_release);
	}
	m_wake.notify_one();
	Signal();
	if (!m_thread.joinable()) {
		m_thread = std::thread([this] { Run(); });
	}
	m_stats.restarts.fetch_add(1, std::memory_order_relaxed);
}

SpeculatedDraw* DrawSpeculator::Take(const uint32_t* packet) {
	auto&    slot     = m_ring[m_gpu_seq % RingSize];
	uint32_t expected = Ready;
	if (!slot.state.compare_exchange_strong(expected, Taken, std::memory_order_acq_rel)) {
		m_stats.missed.fetch_add(1, std::memory_order_relaxed);
		return nullptr;
	}
	if (slot.epoch != m_epoch.load(std::memory_order_relaxed) || slot.seq != m_gpu_seq) {
		slot.state.store(Free, std::memory_order_release);
		m_stats.missed.fetch_add(1, std::memory_order_relaxed);
		return nullptr;
	}
	if (slot.packet != packet) {
		// The walk lost step with the GPU thread's draws: start it again.
		slot.state.store(Free, std::memory_order_release);
		m_mismatch = true;
		m_stats.mismatched.fetch_add(1, std::memory_order_relaxed);
		return nullptr;
	}
	m_stats.taken.fetch_add(1, std::memory_order_relaxed);
	slot.draw.reads_current = ReadsCurrent(slot.draw);
	return &slot.draw;
}

// Draws of one BDA epoch need not see CPU writes made during it (see RenderContext::PrepareBda):
// when the speculation read guest memory in the current epoch, what it read is what the draw may
// read, as for stream copies kept within an epoch (BufferCache::ObtainBuffer). GPU writes are not
// CPU writes: the reads must also have been made after the last buffer or image became
// GPU-written, since the readers refuse (or read around) GPU-written memory. A write counted at
// the Release of draw k happened before the walk read draw k's release count only if k is lower
// than it. KYTY_DEBUG_AB=specreads makes the reads again in every other window.
bool DrawSpeculator::ReadsCurrent(const SpeculatedDraw& draw) {
	static const bool ab = AbSelected("specreads");
	if (ab && AbFeatureOff()) {
		return false;
	}
	const auto epoch = m_renderer.CurrentBdaEpoch();
	return epoch != 0 && draw.read_epoch == epoch && m_last_write_seq < draw.read_after &&
	       m_renderer.GetBufferCache().GpuDirtyGeneration() == m_seen_buffer_writes &&
	       m_renderer.GetTextureCache().GpuModifiedGeneration() == m_seen_image_writes;
}

void DrawSpeculator::Release(SpeculatedDraw* draw) {
	if (draw != nullptr) {
		m_ring[m_gpu_seq % RingSize].state.store(Free, std::memory_order_release);
	}
	const auto buffer_writes = m_renderer.GetBufferCache().GpuDirtyGeneration();
	const auto image_writes  = m_renderer.GetTextureCache().GpuModifiedGeneration();
	if (buffer_writes != m_seen_buffer_writes || image_writes != m_seen_image_writes) {
		m_seen_buffer_writes = buffer_writes;
		m_seen_image_writes  = image_writes;
		m_last_write_seq     = m_gpu_seq;
	}
	m_gpu_seq++;
	m_gpu_next.store(m_gpu_seq, std::memory_order_release);
	if (m_waiting.load(std::memory_order_acquire) &&
	    m_gpu_seq >= m_wake_at.load(std::memory_order_acquire)) {
		Signal();
	}
	static const bool stats = EnvEnabled("KYTY_DEBUG_SPEC_STATS");
	if (stats && m_gpu_seq % 100000u == 0) {
		PrintStats();
	}
}

bool DrawSpeculator::RestartDue() const {
	const auto stopped = m_stopped_seq.load(std::memory_order_acquire);
	return m_mismatch || (stopped != NotStopped && stopped < m_gpu_seq);
}

void DrawSpeculator::Stop(uint64_t seq, uint64_t epoch, uint32_t opcode) {
	m_stats.stops[opcode & 0xffu].fetch_add(1, std::memory_order_relaxed);
	if (!Interrupted(epoch)) {
		m_stopped_seq.store(seq, std::memory_order_release);
	}
}

void DrawSpeculator::PrintStats() {
	std::string stops;
	std::array<std::pair<uint64_t, uint32_t>, 256> ranked {};
	for (uint32_t opcode = 0; opcode < 256; opcode++) {
		ranked[opcode] = {m_stats.stops[opcode].load(std::memory_order_relaxed), opcode};
	}
	std::ranges::sort(ranked, std::greater {});
	for (uint32_t i = 0; i < 6 && ranked[i].first != 0; i++) {
		char text[32];
		std::snprintf(text, sizeof(text), " %02x:%" PRIu64, ranked[i].second, ranked[i].first);
		stops += text;
	}
	const auto failures = m_renderer.GetPipelineCache().SpeculationFailures();
	std::printf("draw-speculator: draws=%" PRIu64 " restarts=%" PRIu64 " walked=%" PRIu64
	            " speculated=%" PRIu64 " behind=%" PRIu64 " taken=%" PRIu64 " missed=%" PRIu64
	            " mismatched=%" PRIu64 " deferred=%" PRIu64 " (unread %" PRIu64 ", unknown %" PRIu64
	            ") stops:%s failures: predict=%" PRIu64
	            " source=%" PRIu64 " unready=%" PRIu64 " refresh=%" PRIu64
	            " (context %zu, shader %zu bytes)\n",
	            m_gpu_seq, m_stats.restarts.load(), m_stats.walked.load(),
	            m_stats.speculated.load(), m_stats.behind.load(), m_stats.taken.load(),
	            m_stats.missed.load(), m_stats.mismatched.load(), m_stats.deferred.load(),
	            m_stats.table_unread.load(), m_stats.table_unknown.load(), stops.c_str(), failures[0], failures[1], failures[2], failures[3],
	            sizeof(HW::Context), sizeof(HW::Shader));
}

DrawSpeculator::Slot* DrawSpeculator::AcquireSlot(uint64_t seq, uint64_t epoch) {
	auto& slot = m_ring[seq % RingSize];
	for (;;) {
		if (Interrupted(epoch)) {
			return nullptr;
		}
		auto state = slot.state.load(std::memory_order_acquire);
		if (state == Free) {
			if (slot.state.compare_exchange_strong(state, Writing, std::memory_order_acq_rel)) {
				return &slot;
			}
			continue;
		}
		// A speculation the GPU thread has passed or will never take.
		if (state == Ready && (slot.epoch != epoch ||
		                       slot.seq < m_gpu_next.load(std::memory_order_acquire))) {
			slot.state.compare_exchange_strong(state, Free, std::memory_order_acq_rel);
			continue;
		}
		// The ring is a full lap ahead: wait until the GPU thread has freed half of it.
		const auto signal = m_signal.load(std::memory_order_acquire);
		m_wake_at.store(seq >= RingSize / 2u ? seq - RingSize / 2u : 0u, std::memory_order_relaxed);
		m_waiting.store(true, std::memory_order_seq_cst);
		const auto again = slot.state.load(std::memory_order_acquire);
		if (again == state && !(again == Ready && slot.seq < m_gpu_next.load()) &&
		    !Interrupted(epoch)) {
			m_signal.wait(signal, std::memory_order_acquire);
		}
		m_waiting.store(false, std::memory_order_relaxed);
	}
}

bool DrawSpeculator::WaitForGpu(uint64_t seq, uint64_t epoch) {
	for (;;) {
		if (Interrupted(epoch)) {
			return false;
		}
		if (m_gpu_next.load(std::memory_order_acquire) > seq) {
			return true;
		}
		const auto signal = m_signal.load(std::memory_order_acquire);
		m_wake_at.store(seq + 1u, std::memory_order_relaxed);
		m_waiting.store(true, std::memory_order_seq_cst);
		if (m_gpu_next.load(std::memory_order_acquire) <= seq && !Interrupted(epoch)) {
			m_signal.wait(signal, std::memory_order_acquire);
		}
		m_waiting.store(false, std::memory_order_relaxed);
	}
}

void DrawSpeculator::Run() {
	KYTY_PROFILER_THREAD("DrawSpeculator");
	ShaderRecompiler::IR::t_resource_scratch_slot = 1;
	auto shadow   = std::make_unique<CommandProcessor>(m_renderer, m_interrupt_event_id);
	auto snapshot = std::vector<Cursor> {};
	for (;;) {
		uint64_t epoch = 0;
		uint64_t seq   = 0;
		// The GPU thread restarts the walk when it reaches the packet the walk stopped at, a few
		// hundred microseconds later at most (the walk runs up to a ring of draws ahead). Waking
		// from a sleep took about as long as the GPU thread needs for a dozen draws, which it then
		// prepares itself (the walk counts them behind): spin for the restart first.
		// KYTY_DEBUG_AB=specspin sleeps at once in every other window.
		static const bool spin_ab = AbSelected("specspin");
		if (!(spin_ab && AbFeatureOff())) {
			constexpr auto SpinTime = std::chrono::microseconds(500);
			const auto     deadline = std::chrono::steady_clock::now() + SpinTime;
			for (uint32_t i = 1; !m_restart_pending.load(std::memory_order_acquire) &&
			                     !m_quitting.load(std::memory_order_acquire);
			     i++) {
				_mm_pause();
				if (i % 64 == 0 && std::chrono::steady_clock::now() >= deadline) {
					break;
				}
			}
		}
		{
			std::unique_lock lock(m_mutex);
			m_wake.wait(lock, [&] { return m_restart || m_quit; });
			if (m_quit) {
				return;
			}
			m_restart = false;
			m_restart_pending.store(false, std::memory_order_relaxed);
			std::swap(m_snapshot, m_spare);
		}
		// The GPU thread fills the other snapshot meanwhile.
		const auto& taken              = *m_spare;
		shadow->m_ctx                  = taken.ctx;
		shadow->m_context_state_pushed = taken.context_state_pushed;
		if (taken.context_state_pushed) {
			shadow->m_saved_ctx = taken.saved_ctx;
		}
		shadow->m_ucfg             = taken.ucfg;
		shadow->m_sh_ctx           = taken.sh_ctx;
		shadow->m_user_data_marker = taken.user_data_marker;
		snapshot                   = taken.stack;
		epoch                      = taken.epoch;
		seq                        = taken.seq;
		Walk(*shadow, std::move(snapshot), epoch, seq);
		snapshot = {};
	}
}

void DrawSpeculator::Walk(CommandProcessor& shadow, std::vector<Cursor> stack, uint64_t epoch,
                          uint64_t seq) {
	auto&              pipelines  = m_renderer.GetPipelineCache();
	const auto&        buffers    = m_renderer.GetBufferCache();
	constexpr uint32_t MaxPackets = 1u << 20u;
	for (uint32_t packets = 0; !stack.empty() && packets < MaxPackets; packets++) {
		if (Interrupted(epoch)) {
			return;
		}
		auto& cursor = stack.back();
		if (cursor.offset_dw >= cursor.commands.size()) {
			stack.pop_back();
			continue;
		}
		const auto* packet    = cursor.commands.data() + cursor.offset_dw;
		const auto  total_dw  = static_cast<uint32_t>(cursor.commands.size());
		const auto  remaining = total_dw - cursor.offset_dw;
		const auto  header    = packet[0];
		if (header == 0x80000000u) {
			cursor.offset_dw++;
			continue;
		}
		const auto packet_dw = KYTY_PM4_LEN(header);
		const auto opcode    = (header >> 8u) & 0xffu;
		// Predicated packets may be skipped, which the walk cannot know.
		if ((header >> 30u) != 3u || remaining < 2 || packet_dw > remaining || (header & 1u) != 0) {
			Stop(seq, epoch, opcode);
			return;
		}
		switch (opcode) {
			case Pm4::IT_SET_CONTEXT_REG:
			case Pm4::IT_SET_SH_REG:
			case Pm4::IT_SET_UCONFIG_REG:
			case Pm4::IT_SET_UCONFIG_REG_INDEX: {
				// The handlers the GPU thread runs, on the shadow's registers.
				const auto handled =
				    g_cp_op_func[opcode](shadow, header & ~1u, packet + 1, remaining, total_dw) + 1;
				if (handled != packet_dw) {
					Stop(seq, epoch, opcode);
					return;
				}
				break;
			}
			case Pm4::IT_SET_CONTEXT_REG_INDIRECT:
			case Pm4::IT_SET_SH_REG_INDIRECT:
			case Pm4::IT_SET_UCONFIG_REG_INDIRECT: {
				// The handler reads (register, value) pairs at the packet's address. Hand it a copy
				// read from the backing, since the guest address can fault on this thread.
				if (packet_dw != 5u) {
					Stop(seq, epoch, opcode);
					return;
				}
				const auto address = (static_cast<uint64_t>(packet[1]) & 0xfffffffcu) |
				                     (static_cast<uint64_t>(packet[2]) << 32u);
				const auto count   = packet[4] & 0x3fffu;
				if (count == 0) {
					break;
				}
				thread_local std::vector<uint32_t> pairs;
				pairs.resize(size_t {count} * 2u);
				const auto bytes = pairs.size() * sizeof(uint32_t);
				const auto read  = [&] {
					if (address == 0) {
						m_stats.table_unread.fetch_add(1, std::memory_order_relaxed);
						return false;
					}
					// Tables in memory without a backing are read at their guest address, as the
					// handler reads them, unless the GPU may have written the pages (a stale hint
					// only faults and reads back, as a guest thread's read would).
					if (!LibKernel::Memory::TryReadBacking(address, pairs.data(), bytes)) {
						if (buffers.IsPageGpuDirtyHint(address) ||
						    buffers.IsPageGpuDirtyHint(address + bytes - 1u)) {
							m_stats.table_unread.fetch_add(1, std::memory_order_relaxed);
							return false;
						}
						std::memcpy(pairs.data(), reinterpret_cast<const void*>(address), bytes);
					}
					if (!IndirectRegistersKnown(opcode, pairs)) {
						m_stats.table_unknown.fetch_add(1, std::memory_order_relaxed);
						return false;
					}
					return true;
				};
				if (!read()) {
					// The stream (or the constant engine) writes the table later. Once the GPU
					// thread has run the next draw, it has read the table: read it then.
					m_stats.deferred.fetch_add(1, std::memory_order_relaxed);
					if (!WaitForGpu(seq, epoch)) {
						return;
					}
					if (!read()) {
						Stop(seq, epoch, opcode);
						return;
					}
				}
				const auto                    copy = reinterpret_cast<uint64_t>(pairs.data());
				const std::array<uint32_t, 4> body {static_cast<uint32_t>(copy),
				                                    static_cast<uint32_t>(copy >> 32u), packet[3],
				                                    packet[4]};
				(void)g_cp_op_func[opcode](shadow, header & ~1u, body.data(), packet_dw, packet_dw);
				break;
			}
			case Pm4::IT_CLEAR_STATE: shadow.m_ctx.Reset(); break;
			case Pm4::IT_NOP: {
				const auto r = KYTY_PM4_R(header);
				if (r == Pm4::R_ZERO && packet_dw >= 2 && (packet[1] & 0xffff0000u) == 0x68750000u) {
					// User data markers classify the SGPR writes that follow (see CpOpMarker).
					const auto id = packet[1] & 0xfffu;
					if (id == 0x4u) {
						shadow.SetUserDataMarker(HW::UserSgprType::Vsharp);
					} else if (id == 0xdu) {
						shadow.SetUserDataMarker(HW::UserSgprType::Region);
					}
				} else if (r == Pm4::R_CONTEXT_STATE && packet_dw >= 3) {
					const auto operation = packet[1];
					const bool push = operation == static_cast<uint32_t>(ContextStateOperation::Push) ||
					                  operation == static_cast<uint32_t>(ContextStateOperation::PushClear);
					const bool pop = operation == static_cast<uint32_t>(ContextStateOperation::Pop);
					if (operation > static_cast<uint32_t>(ContextStateOperation::PushClear) ||
					    (push && shadow.m_context_state_pushed) ||
					    (pop && !shadow.m_context_state_pushed)) {
						Stop(seq, epoch, opcode);
						return;
					}
					shadow.ApplyContextStateOperation(static_cast<ContextStateOperation>(operation));
				} else if (r == Pm4::R_DISPATCH_RESET) {
					shadow.Reset();
				}
				break;
			}
			case Pm4::IT_INDIRECT_BUFFER: {
				// Calls and chains are followed; branches (the 14-dword form) end the walk.
				if (packet_dw != 4u) {
					Stop(seq, epoch, opcode);
					return;
				}
				const auto* address = reinterpret_cast<const uint32_t*>(
				    packet[1] | (static_cast<uint64_t>(packet[2]) << 32u));
				const auto size  = packet[3] & 0xfffffu;
				const bool chain = (packet[3] & (1u << 20u)) != 0;
				cursor.offset_dw += packet_dw;
				if (size == 0) {
					continue;
				}
				if (address == nullptr) {
					Stop(seq, epoch, opcode);
					return;
				}
				const std::span<const uint32_t> commands(address, size);
				if (chain) {
					stack.back() = {commands};
				} else {
					stack.push_back({commands});
				}
				continue;
			}
			default:
				if (IsDrawPacket(opcode)) {
					// Preparing a draw takes the walk about as long as the GPU thread takes for one,
					// so a draw the GPU thread is about to reach is lost to it: skip such draws, and
					// the walk gets ahead after a restart instead of racing for the next draw each
					// time. KYTY_DEBUG_AB=speclead races for every draw not yet reached in every
					// other window.
					static const bool lead_ab = AbSelected("speclead");
					const uint64_t    lead    = lead_ab && AbFeatureOff() ? 0u : SpeculationLead;
					if (seq < m_gpu_next.load(std::memory_order_acquire) + lead) {
						// Behind the GPU thread, or too close to it: not worth speculating.
						m_stats.behind.fetch_add(1, std::memory_order_relaxed);
					} else if (opcode != Pm4::IT_DISPATCH_DRAW_PREAMBLE) {
						auto* slot = AcquireSlot(seq, epoch);
						if (slot == nullptr) {
							return;
						}
						m_stats.walked.fetch_add(1, std::memory_order_relaxed);
						// Before any read (acquire loads): see ReadsCurrent.
						slot->draw.read_epoch = m_renderer.CurrentBdaEpoch();
						slot->draw.read_after = m_gpu_next.load(std::memory_order_acquire);
						if (pipelines.SpeculateGraphicsPrograms(shadow.m_ctx, shadow.m_sh_ctx,
						                                        shadow.m_ucfg, buffers, slot->draw)) {
							slot->epoch  = epoch;
							slot->seq    = seq;
							slot->packet = packet;
							slot->state.store(Ready, std::memory_order_release);
							m_stats.speculated.fetch_add(1, std::memory_order_relaxed);
						} else {
							slot->state.store(Free, std::memory_order_release);
						}
					}
					seq++;
				} else if (!PassesRegisters(opcode)) {
					Stop(seq, epoch, opcode);
					return;
				}
				break;
		}
		cursor.offset_dw += packet_dw;
	}
}

} // namespace Libs::Graphics
