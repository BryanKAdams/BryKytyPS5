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

// KYTY_DEBUG_DRAW_PHASES=<pixel shader hash> times the render thread's CPU phases of that pixel
// shader's draws and prints their average every 5 s. Each Mark charges the time since the previous
// mark to its phase; marks outside a Begin/End pair (other threads, other work) do nothing.
struct DrawPhaseTimer {
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
	static uint64_t Hash();
	void            Begin() {
		if (Hash() != 0) [[unlikely]] {
			active = true;
			current.fill(0);
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
	static uint64_t Now();

	bool                         active = false;
	uint64_t                     last   = 0;
	std::array<uint64_t, Count>  current {};
};
inline thread_local DrawPhaseTimer g_draw_phases;

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
