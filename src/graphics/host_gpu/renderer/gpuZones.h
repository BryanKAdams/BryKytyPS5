#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUZONES_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUZONES_H_

#include "graphics/host_gpu/renderer/drainStats.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>

// Sites that record GPU work name its zone (see DrainStats::Zone). With zones off a mark costs
// one load; with them on, the render scheduler installs the marker and timestamps each change
// of zone in its recording buffer. Marks on other command buffers are ignored.
namespace Libs::Graphics::GpuZones {

using Marker = void (*)(void* context, vk::CommandBuffer buffer, DrainStats::Zone zone,
                        uint64_t key, uint64_t pixels);

inline Marker g_marker  = nullptr;
inline void*  g_context = nullptr;

// The key for marks that pass none: the texture cache sets the guest address of the image it
// transfers, with DownloadKey set for transfers back to guest memory.
inline thread_local uint64_t t_key       = 0;
constexpr uint64_t           DownloadKey = uint64_t {1} << 63u;

class KeyScope final {
public:
	explicit KeyScope(uint64_t key) noexcept: m_previous(t_key) { t_key = key; }
	~KeyScope() { t_key = m_previous; }
	KeyScope(const KeyScope&)            = delete;
	KeyScope& operator=(const KeyScope&) = delete;

private:
	uint64_t m_previous;
};

[[nodiscard]] inline bool Enabled() {
	return g_marker != nullptr;
}

// `pixels` is a draw's render area; per-pixel costs stay comparable when the game's dynamic
// resolution changes between runs.
inline void Mark(vk::CommandBuffer buffer, DrainStats::Zone zone, uint64_t key = 0,
                 uint64_t pixels = 0) {
	if (g_marker != nullptr) [[unlikely]] {
		g_marker(g_context, buffer, zone, key != 0 ? key : t_key, pixels);
	}
}

} // namespace Libs::Graphics::GpuZones

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUZONES_H_
