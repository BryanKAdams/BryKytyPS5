#ifndef KYTY_COMMON_EMULATOR_CONFIG_H_
#define KYTY_COMMON_EMULATOR_CONFIG_H_

#include "common/common.h"

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace Config {

void Initialize();
void Shutdown();

struct Lifecycle {
	static constexpr const char* name       = "Config";
	static constexpr auto        initialize = Config::Initialize;
	static constexpr auto        shutdown   = Config::Shutdown;
};

enum class ShaderOptimizationType { None, Size, Performance };

enum class LogDirection { Silent, Console, File };

enum class PresentMode { Fifo, Mailbox, Immediate };

enum class BdaSyncMode { Selective, Legacy, SelectiveChecked };

using Keymap = std::vector<std::string>;

constexpr uint32_t DEFAULT_CONSOLE_LANGUAGE = 1;
constexpr uint32_t MAX_CONSOLE_LANGUAGE     = 29;
constexpr std::size_t MAX_USER_NAME_LENGTH = 16;
constexpr int32_t DEFAULT_USER_ID           = 1000;

constexpr bool IsConfiguredUserIdValid(int32_t user_id) {
	constexpr int32_t USER_ID_EVERYONE = 0xfe;
	constexpr int32_t USER_ID_SYSTEM   = 0xff;
	return user_id >= 0 && user_id != USER_ID_EVERYONE && user_id != USER_ID_SYSTEM;
}

struct ConfigOptions {
	uint32_t               screen_width                = 1280;
	uint32_t               screen_height               = 720;
	std::string            user_name                   = "Kyty";
	int32_t                user_id                     = DEFAULT_USER_ID;
	std::string            audio_input_device;
	PresentMode            present_mode                = PresentMode::Mailbox;
	BdaSyncMode            bda_sync_mode                   = BdaSyncMode::Selective;
	int32_t                gpu_index                   = -1;
	bool                   fullscreen_enabled          = false;
	bool                   vr_enabled                  = false;
	bool                   amd_cpu_enabled             = false;
	uint32_t               vblank_frequency            = 60;
	uint32_t               console_language            = DEFAULT_CONSOLE_LANGUAGE;
	bool                   vulkan_validation_enabled   = false;
	bool                   shader_validation_enabled   = false;
	bool                   shader_precompile_enabled       = true;
	ShaderOptimizationType shader_optimization_type    = ShaderOptimizationType::None;
	LogDirection           shader_log_direction        = LogDirection::Silent;
	std::filesystem::path  shader_log_folder           = "_Shaders";
	bool                   command_buffer_dump_enabled = false;
	std::filesystem::path  command_buffer_dump_folder  = "_Buffers";
	bool                   graphics_debug_dump_enabled = false;
	LogDirection           printf_direction            = LogDirection::Silent;
	std::filesystem::path  printf_output_file          = "_kyty.txt";
	bool                   profiler_enabled            = false;
	bool                   spirv_debug_printf_enabled  = false;
	bool                   gpu_assisted_validation_enabled = false;
	bool                   renderdoc_enabled           = false;
	bool                   readback_linear_images      = false;
	bool                   tessellation_enabled        = false;
	bool                   playgo_hack_enabled         = false;
	uint32_t               drain_stats_interval        = 0;
	bool                   dcc_gpu_clear_enabled       = true;
	bool                   async_submit_enabled        = true;
	bool                   gpu_mesh_indirect_enabled   = true;
	uint32_t               gpu_frames_ahead            = 0;
	uint32_t               label_flush_interval_us     = 2000;
	uint32_t               gpu_timestamp_scale_percent = 100;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	bool red_zone_protection_enabled = false;
#endif
	Keymap keymap;
};

void Load(const ConfigOptions& cfg);

uint32_t GetScreenWidth();
uint32_t GetScreenHeight();
const std::string& GetUserName();
int32_t  GetUserId();
const std::string& GetAudioInputDevice();
PresentMode GetPresentMode();
BdaSyncMode        GetBdaSyncMode();
int32_t GetGpuIndex();
bool     FullscreenEnabled();
bool     VrEnabled();
bool     AmdCpuEnabled();
uint32_t GetVblankFrequency();
uint32_t GetConsoleLanguage();
bool     VulkanValidationEnabled();

bool                   ShaderValidationEnabled();
bool                   ShaderPrecompileEnabled();
ShaderOptimizationType GetShaderOptimizationType();
LogDirection           GetShaderLogDirection();
std::filesystem::path  GetShaderLogFolder();

bool                  CommandBufferDumpEnabled();
std::filesystem::path GetCommandBufferDumpFolder();

bool GraphicsDebugDumpEnabled();

LogDirection          GetPrintfDirection();
std::filesystem::path GetPrintfOutputFile();

bool ProfilerEnabled();

bool SpirvDebugPrintfEnabled();

bool GpuAssistedValidationEnabled();

bool RenderDocEnabled();
bool ReadbackLinearImagesEnabled();
bool TessellationEnabled();
bool PlayGoHackEnabled();
// Seconds between GPU wait reports; 0 disables the accounting.
uint32_t GetDrainStatsInterval();
// Apply GPU-written DCC fast clears on the GPU instead of reading the keys back.
bool DccGpuClearEnabled();
// Submit the GPU thread's command buffers from a dedicated queue thread.
bool AsyncSubmitEnabled();
// Build mesh-emulated indirect draws with GPU-written arguments on the GPU.
bool GpuMeshIndirectEnabled();
// Frames the game may build ahead of Thread_Gpu at sceAgcSuspendPoint; 0 waits for idle.
uint32_t GetGpuFramesAhead();
// Minimum time between submits made at plain RELEASE_MEM labels; 0 submits at every idle label.
uint32_t GetLabelFlushIntervalUs();
// Stretch time measured between guest GPU timestamps within a frame, in percent (100 = off).
// Games that size their dynamic resolution from GPU timestamps then leave more GPU headroom.
uint32_t GetGpuTimestampScalePercent();
// Changes it while running (the settings panel); takes effect at the next timestamp.
void     SetGpuTimestampScalePercent(uint32_t percent);
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
bool RedZoneProtectionEnabled();
#endif

const Keymap& GetKeymap();

} // namespace Config

#endif /* KYTY_COMMON_EMULATOR_CONFIG_H_ */
