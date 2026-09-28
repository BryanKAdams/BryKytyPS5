#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DEBUG_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DEBUG_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <string>

namespace Libs::Graphics {

// Counted for KYTY_DEBUG_DRAW_LOG: dynamic rendering instances begun and ended, and image
// barriers recorded.
struct RenderDebugCounters {
	std::atomic<uint64_t> render_begins {0};
	std::atomic<uint64_t> render_ends {0};
	std::atomic<uint64_t> image_barriers {0};
	std::atomic<bool>     counting {false};     // Set while draws are logged or summarized.
	std::atomic<bool>     log_barriers {false}; // Set while a logged draw records.
};
inline RenderDebugCounters g_render_debug_counters;

// With KYTY_DEBUG_DRAW_PHASES set, KYTY_DEBUG_AB=<features> turns the named features off in every
// other 5 s window, so one run compares both under the same scene and machine load; each
// draw-phases line reports its window's mode. Features: reuse (texture and render-target reuse;
// also KYTY_DEBUG_AB=1), pending (rate-limited GPU progress queries at draw entry), backing
// (lock-free cached guest backing reads), bdaepoch (one BDA synchronization per epoch), scratch
// (pooled tiler scratch buffers), cleanpages (remembered clean pages for GPU-write queries),
// cpwrite (BDA epochs started by the GPU thread's own guest memory writes), streamhost (the stream
// ring in cached host memory rather than device memory), colorclear (skipped color clear
// rechecks), prioritywait (draw entry leaving operations queued rather than waiting for the
// priority thread).
[[nodiscard]] bool AbSelected(const char* feature) noexcept;
[[nodiscard]] bool AbFeatureOff() noexcept;
// GPU busy time as drain stats measure it (--drain-stats), for the draw-phases line's gpu-ms/s.
inline std::atomic<uint64_t> g_gpu_busy_ns {0};
// GPU buffers and images created and destroyed, for the draw-phases line's allocation churn.
struct AllocationCounters {
	std::atomic<uint64_t> buffers_created {0};
	std::atomic<uint64_t> buffers_destroyed {0};
	std::atomic<uint64_t> images_created {0};
	std::atomic<uint64_t> images_destroyed {0};
};
inline AllocationCounters g_allocation_counters;

// KYTY_DEBUG_DRAW_PHASES=<pixel shader hash> times the render thread's CPU phases of that pixel
// shader's draws and prints their average every 5 s; =all times every draw. Each Mark charges the
// time since the previous mark to its phase; marks outside a Begin/End pair (other threads, other
// work) do nothing.
struct DrawPhaseTimer {
	static constexpr uint64_t AllDraws = ~uint64_t {0};
	enum Phase : uint32_t {
		Setup,            // Draw entry up to shader lookup.
		VertexParams,     // Vertex stage registers and code hash.
		PixelParams,      // Pixel stage registers and code hash.
		PixelProgram,     // Pixel program lookup and resource materialization.
		VertexProgram,    // Vertex program lookup and resource materialization.
		Targets,          // Render target resolution.
		StageTextures,    // PrepareBindings: textures.
		StageSamplers,    // PrepareBindings: samplers.
		StageBindings,    // PrepareBindings: user data.
		FindBuffers,      // PrepareGraphicsBindings: buffer discovery.
		RebindImages,     // PrepareGraphicsBindings: image views.
		BufferViews,      // PrepareGraphicsBindings: storage buffer binding.
		GraphicsBindings, // PrepareGraphicsBindings: uploads and the rest.
		RenderTargets,    // AcquireRenderTargets.
		Pipeline,         // Pipeline lookup.
		Records,          // Mesh draw records.
		Commit,           // CommitBindings.
		Record,           // Dynamic state, rendering and draw commands.
		Tail,             // After the draw.
		Count
	};
	// Spans timed on their own inside the phases; a probe overlaps its phase's time.
	enum Probe : uint32_t {
		TargetImage,      // Render-target FindImage.
		TextureImage,     // Texture FindImage.
		TextureDescribe,  // Texture description lookup and copy.
		BufferWritten,    // ObtainBuffer for written storage buffers.
		BufferRead,       // ObtainBuffer for read-only storage buffers.
		BufferInvalidate, // Texture invalidation behind written storage buffers.
		Upload,           // Flattened SRT and shader data uploads.
		FindFinish,       // FindImage and RefindImage after the lookup (DCC clear checks).
		StreamCopy,       // ObtainBuffer's copies of small CPU-written buffers.
		PendingOps,       // The scheduler's completed operations run at draw entry.
		Bda,              // PrepareBda for stages reading memory through addresses.
		SpeculationReads, // Repeating a speculated stage's guest reads before adopting it.
		ResourceRefresh,  // Refreshing a stage's resources that were not adopted.
		ProgramMatch,     // Finding the refreshed stage's permutation.
		InputsCheck,      // Comparing the registers of speculation-prepared stage inputs.
		ProbeCount
	};
	class ProbeScope {
	public:
		ProbeScope(DrawPhaseTimer& timer, Probe probe)
		    : m_timer(timer.active ? &timer : nullptr), m_probe(probe),
		      m_start(m_timer != nullptr ? Now() : 0) {}
		~ProbeScope() {
			if (m_timer != nullptr) [[unlikely]] {
				m_timer->probes[m_probe] += Now() - m_start;
			}
		}
		ProbeScope(const ProbeScope&)            = delete;
		ProbeScope& operator=(const ProbeScope&) = delete;

	private:
		DrawPhaseTimer* m_timer;
		Probe           m_probe;
		uint64_t        m_start;
	};
	static uint64_t Hash();
	// The time stamp counter the phases are measured in.
	static uint64_t Now();
	void            Begin() {
		if (Hash() != 0) [[unlikely]] {
			active = true;
			current.fill(0);
			probes.fill(0);
			last = Now();
		}
	}
	void Mark(Phase phase) {
		if (active) [[unlikely]] {
			const auto now = Now();
			current[phase] += now - last;
			last = now;
		}
	}
	// Accounts the draw if its pixel shader is the one timed.
	void End(uint64_t pixel_hash);

private:
	bool                             active = false;
	uint64_t                         last   = 0;
	std::array<uint64_t, Count>      current {};
	std::array<uint64_t, ProbeCount> probes {};
};
inline thread_local DrawPhaseTimer g_draw_phases;

// With KYTY_DEBUG_DRAW_PHASES: the render thread's time in each PM4 packet's handler, by opcode,
// and in ProcessPm4 between handlers, printed as a "pm4-ops" line after each draw-phases line.
// Draw handlers include their draw phases; the rest is the time no draw phase covers.
struct Pm4OpTimer {
	// How a draw packet's handler ended (set by the handler, reset before each packet).
	enum Outcome : uint8_t {
		Drawn,
		Empty,         // No vertices or instances.
		MetadataOp,    // A color metadata (clear) operation instead of a draw.
		DepthCopy,     // A depth/stencil copy instead of a draw.
		Resolve,       // A color resolve instead of a draw.
		NoShader,      // No valid vertex shader or no topology.
		NotPrepared,   // Programs pending or no target (PrepareDrawRenderState).
		RectListSkip,  // A rect list with nothing to interpolate.
		OutcomeCount
	};
	std::array<uint64_t, 256> ticks {};
	std::array<uint64_t, 256> counts {};
	uint64_t                  between = 0;
	Outcome                   outcome = Drawn;
	// By outcome: [0] indexed draw packets, [1] DRAW_INDEX_AUTO.
	std::array<std::array<uint64_t, OutcomeCount>, 2> outcome_ticks {};
	std::array<std::array<uint64_t, OutcomeCount>, 2> outcome_counts {};
};
inline thread_local Pm4OpTimer g_pm4_ops;

// KYTY_DEBUG_UPLOADS=1 (with KYTY_DEBUG_DRAW_PHASES): guest memory copied for the GPU, counted
// per source and printed after each draw-phases line with the images and 4 MiB regions copied
// most. A copy counts under its thread's innermost UploadSourceScope.
enum class UploadSource : uint8_t {
	Buffer, // SynchronizeBuffer outside the scopes below, mostly for bound buffers.
	Bda,    // PrepareBda's dirty-page synchronization.
	Stream, // ObtainBuffer's stream-buffer copies of small CPU-written ranges.
	Image,  // Image uploads (whole image ranges).
	Fault,  // Not a copy: guest write faults on tracked pages, counted as one page each.
	BdaPass, // Not a copy: PrepareBda calls.
	BdaSync, // Not a copy: PrepareBda calls that synchronized (the first of their epoch).
	Count
};
inline thread_local UploadSource t_upload_source = UploadSource::Buffer;
[[nodiscard]] bool UploadStatsEnabled() noexcept;
void               RecordUpload(UploadSource source, uint64_t address, uint64_t bytes) noexcept;
void RecordImageUpload(uint64_t address, uint64_t size, uint32_t width, uint32_t height,
                       uint32_t guest_format, uint32_t tile_mode, bool buffer_modified) noexcept;
class UploadSourceScope {
public:
	explicit UploadSourceScope(UploadSource source) noexcept: m_previous(t_upload_source) {
		t_upload_source = source;
	}
	~UploadSourceScope() { t_upload_source = m_previous; }
	UploadSourceScope(const UploadSourceScope&)            = delete;
	UploadSourceScope& operator=(const UploadSourceScope&) = delete;

private:
	UploadSource m_previous;
};

class CommandBuffer;

namespace HW {
class Context;
class UserConfig;
struct RenderTarget;
struct ScanModeControl;
struct ScreenViewport;
} // namespace HW

struct ScissorRect {
	int left   = 0;
	int top    = 0;
	int right  = 0;
	int bottom = 0;
};

uint32_t                 render_target_mask_slot(uint32_t mask, uint32_t slot);
uint32_t                 render_target_first_bound_slot(const CommandBuffer& buffer);
bool                     graphics_debug_dump_enabled();
void                     uc_print(const char* func, const HW::UserConfig& uc);
void                     uc_check(const HW::UserConfig& uc);
std::string              rt_print(const char* func, const HW::RenderTarget& rt);
bool                     RenderIsColorTileModeLinear(Prospero::TileMode tile_mode);
void                     hw_print(const CommandBuffer& buffer);
void                     hw_check(const CommandBuffer& buffer);
void                     LogDrawPhase(const char* draw_name, const char* phase);
ScissorRect calc_final_scissor(const HW::ScreenViewport& vp, const HW::ScanModeControl& smc,
                               vk::Extent2D extent, uint32_t viewport_index);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DEBUG_H_
