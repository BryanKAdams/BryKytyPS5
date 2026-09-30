#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_WINDOWINTERNAL_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_WINDOWINTERNAL_H_

#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/presentation/frameStatistics.h"

#include <SDL3/SDL.h>
#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

namespace Libs::Graphics {

class Presenter;
class RenderContext;

struct SurfaceCapabilities {
	vk::SurfaceCapabilitiesKHR        capabilities {};
	std::vector<vk::SurfaceFormatKHR> formats;
	std::vector<vk::PresentModeKHR>   present_modes;
};

// A linear blit reads 2x2 source texels per destination pixel, so a blit that shrinks by more
// than 2x skips texels and aliases fine guest detail into a static moire (a 4K game's per-pixel
// dither becomes a diagonal pattern in a 1280x720 window). Prepared frames keep a mip chain: a
// 2:1 linear blit into the next level is an exact 2x2 box filter, and the blit to the swapchain
// starts from the first level that is at most twice the target size.
[[nodiscard]] uint32_t PresentSourceLevel(vk::Extent2D source, vk::Extent2D target,
                                          uint32_t levels) noexcept;
// Fills levels 1..level of a 2D color image from level 0, which must be in eTransferSrcOptimal.
// Levels 1..level are left in eTransferSrcOptimal.
void RecordPresentDownscale(vk::CommandBuffer command, vk::Image image, vk::Extent2D extent,
                            uint32_t level);

// A linear blit that shrinks by less than 2x (a 4K frame in a maximized 2560x1369 window) reads
// texel pairs at uneven phases, so a per-pixel dither survives at up to half strength as a
// diagonal beat. PresentFilter averages each target pixel over a two-texel box instead, which
// cancels one-texel patterns at any phase. An exact 2:1 blit already is that box, so it and
// upscales keep the blit.
[[nodiscard]] bool PresentNeedsFilter(vk::Extent2D source, vk::Extent2D target) noexcept;

class PresentFilter final {
public:
	PresentFilter()  = default;
	~PresentFilter() = default;
	KYTY_CLASS_NO_COPY(PresentFilter);

	// Draws level `level` (extent `source`, at most twice `target` on each axis) of `source_view`
	// over all of `target_view`. The source view covers every level and is in
	// eShaderReadOnlyOptimal; the target is in eColorAttachmentOptimal.
	void Record(vk::Device device, vk::CommandBuffer command, vk::ImageView source_view,
	            uint32_t level, vk::Extent2D source, vk::ImageView target_view,
	            vk::Format target_format, vk::Extent2D target);
	void Release(vk::Device device);

private:
	vk::Format              m_format      = vk::Format::eUndefined;
	vk::Sampler             m_sampler     = nullptr;
	vk::DescriptorSetLayout m_descriptors = nullptr;
	vk::PipelineLayout      m_layout      = nullptr;
	vk::Pipeline            m_pipeline    = nullptr;
};

struct WindowLoopState {
	SDL_Event        event {};
	bool             need_exit = false;
	std::atomic_bool paused    = false;
};

struct WindowContext {
	WindowContext();
	~WindowContext();
	KYTY_CLASS_NO_COPY(WindowContext);

	[[nodiscard]] static vk::PhysicalDeviceVulkan12Features RequiredVulkan12Features() noexcept;
	[[nodiscard]] static vk::PhysicalDeviceVulkan13Features RequiredVulkan13Features() noexcept;
	[[nodiscard]] static uint32_t InitialWindowFlags(bool fullscreen) noexcept;
	void                                                    CreateVulkan();
	void                                                    RecreateSurface();
	void                                                    RefreshSurfaceCapabilities();
	void                                                    UpdateIcon();
	void                                                    UpdateTitle(bool new_frame);
	void                                                    Resize(uint32_t width, uint32_t height);
	void ProcessWindowEvent(const SDL_WindowEvent& event);
	void ProcessDisplayEvent(const SDL_DisplayEvent& event);
	void ProcessEvent(double time_seconds);
	void Run();

	GraphicContext                 graphic_ctx;
	SDL_Window*                    window        = nullptr;
	vk::SurfaceKHR                 surface       = nullptr;
	SurfaceCapabilities            surface_capabilities;
	std::unique_ptr<RenderContext> render_context;
	std::unique_ptr<Presenter>     presenter;
	WindowLoopState                loop;
	FrameStatistics                frame_statistics;

	Common::Mutex mutex;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_WINDOWINTERNAL_H_
