#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAINSTATS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAINSTATS_H_

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>

// Opt-in accounting of host waits on the GPU (--drain-stats). Wait sites record their kind and
// duration under the calling thread's innermost reason scope and the PM4 packet it is executing.
// When disabled, a wait site costs one relaxed load.
namespace Libs::Graphics::DrainStats {

enum class Reason : uint8_t {
	Unattributed,
	GuestReadFault,
	GuestWriteFault,
	GpuThreadReadFault,
	GpuThreadWriteFault,
	KernelInvalidate,
	DccClear,
	Predicate,
	GdsReadback,
	Unmap,
	TexturePendingDownload,
	FreeImage,
	TextureGc,
	BufferGc,
	UploadRingWrap,
	StreamRingWrap,
	DownloadRingWrap,
	FaultBuffer,
	PresentFrame,
	EagerReadback,
	Count,
};

enum class Kind : uint8_t {
	FullDrain,     // Submits the open command buffer and waits for it.
	TickWait,      // Blocks on an already submitted tick.
	PriorityWait,  // Blocks until publication callbacks up to a tick have run.
	BlockedPoll,   // Thread_Gpu idles because every queue is suspended.
	Readback,      // A buffer download was recorded; the value is its size in bytes.
	ReadbackClean, // ReadMemory found nothing to download.
	DccMetaWrite,  // A GPU write covered known DCC metadata; the value is its size in bytes.
	DccCheck,      // A DCC lookup read GPU-written metadata; the value is slices it cleared.
	DccGpuCheck,   // A DCC lookup checked GPU-written metadata on the GPU; the value is slices.
	Submit,        // vkQueueSubmit, including the wait for the queue lock; the value is ns.
	QueueLockWait, // Time a submit waited for the queue lock (held by present or another submit).
	IndirectArgsCpu, // An indirect draw read CPU-clean args; the value counts mesh-emulated draws.
	IndirectArgsGpu, // An indirect draw read GPU-written args; the value counts mesh-emulated draws.
	GpuBusy,       // GPU execution time (union of command-buffer intervals), in ns.
	GpuGap,        // GPU time between command buffers with none executing, in ns.
	OcclusionQuery,     // A ZPASS_DONE occlusion-counter dump.
	OcclusionPredicate, // SET_PREDICATION on occlusion results.
	GpuTimestamp,       // A GPU timestamp write (RELEASE_MEM/EOP or COPY_DATA of the clock).
	ShaderCompile,      // Shader lookup that took at least 1 ms (a new permutation), in ns.
	PipelineCreate,     // Pipeline lookup that took at least 1 ms (a new pipeline), in ns.
	Lookahead,          // A pipeline look-ahead walk, including the shaders it translated, in ns.
	GpuThreadIdle,      // Thread_Gpu waiting for a submission or command, in ns.
	Count,
};

// GPU work zones (KYTY_GPU_ZONES=1 with --drain-stats). The render scheduler timestamps each
// change of zone in its recording buffer and charges the interval up to the next timestamp to
// the zone that opened it. Work recorded without a mark joins the zone before it.
enum class Zone : uint8_t {
	Unmarked,           // From the start of a command buffer to its first mark.
	GameDraw,           // A guest draw, keyed by its pixel shader hash (vertex shader without one).
	GameDrawPredicated, // A GameDraw under SET_PREDICATION, e.g. behind an occlusion query.
	GameDispatch,       // A guest compute dispatch, keyed by its shader hash.
	MeshArgs,           // The mesh indirect-arguments pre-pass.
	Tiler,              // Tiling and detiling compute for texture uploads and downloads.
	DccClear,           // DCC fast-clear resolves.
	FaultBuffer,        // The GPU page-fault buffer scan.
	Blit,               // Blit-helper draws (format conversions and scaled copies).
	ImageCopy,          // Image copies and clears recorded by images and the texture cache.
	BufferCopy,         // Buffer cache and stream-buffer copies, uploads, and downloads.
	Count,
};

struct ZoneSample {
	Zone     zone;
	uint64_t key;
	uint64_t pixels; // Render area of the interval's first draw, or 0.
	uint64_t ns;
};

// PM4 packets are indexed by opcode; IT_NOP packets carrying a Kyty custom code use 256 + code.
constexpr uint32_t Pm4OpCount = 256 + 64;
constexpr uint32_t NoPm4Op    = Pm4OpCount;

inline std::atomic_bool         g_enabled {false};
inline thread_local Reason      t_reason = Reason::Unattributed;
inline thread_local uint32_t    t_pm4_op = NoPm4Op;
// The PM4 packet being executed carries the predication bit.
inline thread_local bool        t_predicated = false;
// Guest instruction that raised the page fault being handled on this thread.
inline thread_local uint64_t    t_fault_pc = 0;

[[nodiscard]] inline bool Enabled() noexcept {
	return g_enabled.load(std::memory_order_relaxed);
}

// Starts the periodic reporter. Records are ignored until this is called.
void Start(uint32_t interval_seconds);
void Stop();

void Record(Kind kind, Reason reason, uint32_t pm4_op, uint64_t value) noexcept;
void CountFrame(bool new_frame) noexcept;
// A GPU-memory fault stalled the faulting thread for `ns`; keyed by the faulting instruction.
void RecordFaultSite(uint64_t pc, uint64_t address, bool write, uint64_t ns) noexcept;
// A recorded GPU command writes [vaddr, vaddr+size). Read fault sites report the newest such
// writer of their address and how many frames ago it was recorded.
void RecordGpuWrite(uint64_t vaddr, uint64_t size) noexcept;
// GPU time of one command buffer's zone intervals.
void RecordZones(const ZoneSample* samples, size_t count) noexcept;

inline void Record(Kind kind, uint64_t value) noexcept {
	if (Enabled()) {
		Record(kind, t_reason, t_pm4_op, value);
	}
}

[[nodiscard]] inline Reason CurrentReason() noexcept {
	return t_reason;
}

[[nodiscard]] inline uint32_t Pm4Op(uint32_t opcode, uint32_t custom) noexcept {
	return opcode == 0x10u && custom < 64u ? 256u + custom : (opcode & 0xffu);
}

inline void SetPm4Op(uint32_t op) noexcept {
	t_pm4_op = op;
}

class ReasonScope final {
public:
	explicit ReasonScope(Reason reason) noexcept: m_previous(t_reason) { t_reason = reason; }
	~ReasonScope() { t_reason = m_previous; }
	ReasonScope(const ReasonScope&)            = delete;
	ReasonScope& operator=(const ReasonScope&) = delete;

private:
	Reason m_previous;
};

class Pm4OpScope final {
public:
	explicit Pm4OpScope(uint32_t op) noexcept: m_previous(t_pm4_op) { t_pm4_op = op; }
	~Pm4OpScope() { t_pm4_op = m_previous; }
	Pm4OpScope(const Pm4OpScope&)            = delete;
	Pm4OpScope& operator=(const Pm4OpScope&) = delete;

private:
	uint32_t m_previous;
};

// Times a lookup that is usually a cache hit and records it only when it took at least 1 ms,
// i.e. it compiled something. Costs one relaxed load when drain stats are off.
class SlowLookupTimer final {
public:
	explicit SlowLookupTimer(Kind kind) noexcept: m_kind(kind), m_enabled(Enabled()) {
		if (m_enabled) {
			m_start = std::chrono::steady_clock::now();
		}
	}
	~SlowLookupTimer() {
		if (m_enabled) {
			const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
			                    std::chrono::steady_clock::now() - m_start)
			                    .count();
			if (ns >= 1000000) {
				Record(m_kind, t_reason, t_pm4_op, static_cast<uint64_t>(ns));
			}
		}
	}
	SlowLookupTimer(const SlowLookupTimer&)            = delete;
	SlowLookupTimer& operator=(const SlowLookupTimer&) = delete;

private:
	std::chrono::steady_clock::time_point m_start {};
	Kind                                  m_kind;
	bool                                  m_enabled;
};

// Times one blocking wait. Construct it only on the path that actually blocks.
class WaitTimer final {
public:
	explicit WaitTimer(Kind kind) noexcept: m_kind(kind), m_enabled(Enabled()) {
		if (m_enabled) {
			m_start = std::chrono::steady_clock::now();
		}
	}
	~WaitTimer() {
		if (m_enabled) {
			const auto elapsed = std::chrono::steady_clock::now() - m_start;
			Record(m_kind, t_reason, t_pm4_op,
			       static_cast<uint64_t>(
			           std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()));
		}
	}
	WaitTimer(const WaitTimer&)            = delete;
	WaitTimer& operator=(const WaitTimer&) = delete;

private:
	std::chrono::steady_clock::time_point m_start {};
	Kind                                  m_kind;
	bool                                  m_enabled;
};

} // namespace Libs::Graphics::DrainStats

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAINSTATS_H_
