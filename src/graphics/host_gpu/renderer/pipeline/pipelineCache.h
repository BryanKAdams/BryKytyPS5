#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/threads.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/shader.h"

#include <atomic>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
struct RenderColorInfo;
struct RenderDepthInfo;
class BufferCache;
class CommandBuffer;
class PipelineLibraryCache;
struct SpeculatedDraw;

namespace ShaderPrecompile {
struct PermutationRecord;
}

namespace HW {
class Context;
class Shader;
class UserConfig;
struct ComputeShaderInfo;
} // namespace HW

// Static feedback flags are ignored by Vulkan when the corresponding dynamic state is enabled.
// Normalize before cache lookup so dynamic draws share a pipeline across feedback aspects.
[[nodiscard]] inline vk::PipelineCreateFlags
AttachmentFeedbackPipelineFlags(vk::ImageAspectFlags aspects, bool dynamic_enabled) {
	if (dynamic_enabled) {
		return {};
	}
	vk::PipelineCreateFlags flags {};
	if (aspects & vk::ImageAspectFlagBits::eColor) {
		flags |= vk::PipelineCreateFlagBits::eColorAttachmentFeedbackLoopEXT;
	}
	if (aspects & (vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil)) {
		flags |= vk::PipelineCreateFlagBits::eDepthStencilAttachmentFeedbackLoopEXT;
	}
	return flags;
}

#pragma pack(push, 1)

struct PipelineStaticParameters {
	bool                       negative_one_to_one      = false;
	bool                       depth_clip_enable        = true;
	vk::PrimitiveTopology      topology                 = vk::PrimitiveTopology::ePointList;
	bool                       primitive_restart_enable = false;
	uint32_t                   samples                  = 1;
	bool                       sample_shading_enable    = false;
	bool                       depth_bounds_test_enable = false;
	float                      depth_min_bounds         = 0.0f;
	float                      depth_max_bounds         = 0.0f;
	uint32_t                   color_mask[RENDER_COLOR_ATTACHMENTS_MAX]           = {};
	bool                       cull_front                                         = false;
	bool                       cull_back                                          = false;
	bool                       face                                               = false;
	bool                       provoking_vtx_last                                 = false;
	vk::PolygonMode            polygon_mode                                       = vk::PolygonMode::eFill;
	uint8_t                    color_srcblend[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    color_comb_fcn[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    color_destblend[RENDER_COLOR_ATTACHMENTS_MAX]      = {};
	uint8_t                    alpha_srcblend[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    alpha_comb_fcn[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    alpha_destblend[RENDER_COLOR_ATTACHMENTS_MAX]      = {};
	bool                       separate_alpha_blend[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	bool                       blend_enable[RENDER_COLOR_ATTACHMENTS_MAX]         = {};
	vk::PipelineCreateFlags    attachment_feedback_loop_flags                     = {};

	bool operator==(const PipelineStaticParameters& other) const noexcept;
};

#pragma pack(pop)

static_assert(std::is_trivially_copyable_v<PipelineStaticParameters>);
static_assert(std::is_standard_layout_v<PipelineStaticParameters>);
static_assert(alignof(PipelineStaticParameters) == 1);
static_assert(sizeof(PipelineStaticParameters) == 129);

struct PipelineRenderingState {
	std::array<vk::Format, RENDER_COLOR_ATTACHMENTS_MAX> color_formats {};
	vk::Format                                           depth_format   = vk::Format::eUndefined;
	vk::Format                                           stencil_format = vk::Format::eUndefined;
	uint32_t                                             color_count    = 0;

	bool operator==(const PipelineRenderingState&) const = default;
};

struct PipelineVertexInputState {
	struct Binding {
		uint32_t stride                           = 0;
		bool     instance                         = false;
		bool     operator==(const Binding&) const = default;
	};
	struct Attribute {
		uint32_t offset                             = 0;
		uint8_t  binding                            = 0;
		bool     operator==(const Attribute&) const = default;
	};

	std::array<Binding, ShaderVertexInputInfo::RES_MAX>   bindings {};
	std::array<Attribute, ShaderVertexInputInfo::RES_MAX> attributes {};
	uint8_t                                               binding_count   = 0;
	uint8_t                                               attribute_count = 0;

	bool operator==(const PipelineVertexInputState&) const = default;
};

// How a shader lookup treats a permutation not compiled yet: compile it now (Wait), or queue
// it on worker threads and return nothing yet, for a draw that may be skipped (Defer, ahead of
// other jobs) or a look-ahead prediction (Prefetch).
enum class ProgramWait { Wait, Defer, Prefetch };

struct ShaderProgram {
	uint64_t         id     = 0;
	vk::ShaderModule module = nullptr;

	explicit operator bool() const { return id != 0 && module != nullptr; }
};

class PipelineCache {
public:
	explicit PipelineCache(GraphicContext& graphics);
	~PipelineCache();
	KYTY_CLASS_NO_COPY(PipelineCache);
	void Save();

	struct Pipeline {
		vk::PipelineLayout      pipeline_layout       = nullptr;
		vk::Pipeline            pipeline              = nullptr;
		// Set 0: a compute shader's descriptors, or a graphics pipeline's vertex-side stages.
		vk::DescriptorSetLayout descriptor_set_layout = nullptr;
		bool                    uses_push_descriptors = false;
		// Set 1 of graphics pipelines: the pixel shader's descriptors. Each set depends on its own
		// stages only, so a stage's descriptor layout is the same in every pipeline that uses it.
		vk::DescriptorSetLayout pixel_set_layout      = nullptr;
		bool                    pixel_uses_push       = false;
		// The push constant range's stages; empty means the stages of the bound shaders.
		vk::ShaderStageFlags    push_constant_stages {};
		// Fast-linked from pipeline libraries, with a link-time-optimized link still to swap in.
		bool                    optimize_pending      = false;
	};

	struct GraphicsPrograms {
		std::array<ShaderProgram, 3> vertex;
		ShaderProgram pixel;
		// A stage is still compiling in the background (ProgramWait::Defer or Prefetch); the
		// other programs may be missing too.
		bool          pending = false;

		[[nodiscard]] uint32_t VertexStageCount() const { return vertex[1] ? 3u : 1u; }
	};

	GraphicsPrograms
	GetGraphicsPrograms(const HW::VertexShaderInfo& vertex_regs,
	                    const HW::PixelShaderInfo& pixel_regs, const HW::ShaderRegisters& sh,
	                    const HW::Context& context, const HW::UserConfig& user_config,
	                    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
	                    bool pixel_active, std::array<ShaderVertexInputInfo, 3>& vertex_info,
	                    ShaderPixelInputInfo& pixel_info, ProgramWait wait = ProgramWait::Wait);
	ShaderProgram GetComputeProgram(const HW::ComputeShaderInfo& regs,
	                                const HW::ShaderRegisters&   sh,
	                                ShaderComputeInputInfo&      input_info,
	                                ProgramWait wait = ProgramWait::Wait, bool* pending = nullptr);

	// With `may_defer` (asynchronous pipelines), a new pipeline whose shader library parts are not
	// compiled yet is not created: its parts are queued on worker threads and the result is null,
	// and the caller skips the draw. Otherwise never null.
	Pipeline* GetGraphicsPipeline(std::span<const RenderColorInfo>       colors,
	                              const RenderDepthInfo&                 depth,
	                              std::span<const ShaderVertexInputInfo> vertex_info,
	                              CommandBuffer& command, const ShaderPixelInputInfo* ps_input_info,
	                              vk::PrimitiveTopology topology, bool primitive_restart_enable,
	                              const GraphicsPrograms& programs,
	                              vk::ImageAspectFlags feedback_aspects = {},
	                              bool                 may_defer        = false);
	Pipeline& GetComputePipeline(const ShaderComputeInputInfo& input_info,
	                             const ShaderProgram&          compute_program);

	// Graphics pipelines created so far; a change across a draw means it compiled one.
	[[nodiscard]] uint64_t GraphicsPipelinesCreated() const noexcept {
		return m_graphics_pipelines_created;
	}
	// Predicts the graphics pipeline a draw with these registers will need, translating its
	// shaders (in the background with ProgramWait::Prefetch, which sets `*pending` while they
	// translate), and queues its missing shader library parts on worker threads. Nothing is recorded
	// or bound; a wrong prediction only costs the compile. Returns the number of parts queued.
	uint32_t PrefetchGraphicsPipeline(const HW::Context& ctx, const HW::Shader& sh,
	                                  const HW::UserConfig& user_config, ProgramWait wait,
	                                  bool* pending);
	// On the draw speculation thread (see DrawSpeculator): refreshes the resources of the vertex
	// (or mesh) and pixel stages a non-tessellated draw with these registers will look up, for the
	// GPU thread to adopt (see SpeculatedDraw). Reads guest memory through the backing only,
	// skipping pages `buffers` hints are GPU-dirty. Stages of program sources not seen yet are
	// left out. Returns whether any stage was refreshed.
	bool SpeculateGraphicsPrograms(const HW::Context& ctx, const HW::Shader& sh,
	                               const HW::UserConfig& user_config, const BufferCache& buffers,
	                               SpeculatedDraw& draw);
	// Why speculated stages failed so far: the draw looks no programs up, the stage's source was
	// never seen, its first refresh is still pending, or the refresh failed.
	[[nodiscard]] std::array<uint64_t, 4> SpeculationFailures() const noexcept;
	// Compute pipelines created so far, for the same purpose.
	[[nodiscard]] uint64_t ComputePipelinesCreated() const noexcept {
		return m_compute_pipelines_created;
	}
	// Predicts the compute pipeline a dispatch with these registers and dispatch initiator will
	// need, translating its shader (like PrefetchGraphicsPipeline), and compiles it on a worker
	// thread; GetComputePipeline takes it over. Returns 1 when a compile was queued.
	uint32_t PrefetchComputePipeline(const HW::Context& ctx, const HW::Shader& sh,
	                                 uint32_t dispatch_initiator, ProgramWait wait, bool* pending);
	// Prints a look-ahead's result with KYTY_PERMUTATION_LOG=1.
	void LogLookahead(uint32_t draws, uint32_t parts) const;
	// Shader translations and module compiles queued or running on worker threads.
	[[nodiscard]] uint32_t BackgroundShaderJobs() const noexcept;
	// How many of those have finished since startup.
	[[nodiscard]] uint64_t BackgroundShaderJobsFinished() const noexcept;

private:
	friend struct AttachmentFeedbackTestAccess;
	struct ProgramCache;

	struct GraphicsPipelineKey {
		PipelineRenderingState   rendering;
		std::array<uint64_t, 3>  vertex_shader_ids {};
		uint64_t                 ps_shader_id = 0;
		PipelineVertexInputState vertex_input;
		PipelineStaticParameters static_params;

		bool operator==(const GraphicsPipelineKey& other) const {
			return rendering == other.rendering && vertex_shader_ids == other.vertex_shader_ids &&
			       ps_shader_id == other.ps_shader_id && vertex_input == other.vertex_input &&
			       static_params == other.static_params;
		}
	};

	struct PipelineKeyHash {
		static void Mix(std::size_t& hash, std::size_t value) {
			hash ^= value + static_cast<std::size_t>(0x9e3779b97f4a7c15ull) + (hash << 6u) +
			        (hash >> 2u);
		}

		// One hash of the packed parameters: mixing their 129 bytes one at a time was a chain of
		// dependent steps on every draw's pipeline lookup.
		static void MixStaticParams(std::size_t& hash, const PipelineStaticParameters& params);

		static void MixRendering(std::size_t& hash, const PipelineRenderingState& rendering) {
			Mix(hash, rendering.color_count);
			for (uint32_t i = 0; i < rendering.color_count; i++) {
				Mix(hash, static_cast<uint32_t>(rendering.color_formats[i]));
			}
			Mix(hash, static_cast<uint32_t>(rendering.depth_format));
			Mix(hash, static_cast<uint32_t>(rendering.stencil_format));
		}
	};

	struct GraphicsPipelineKeyHash {
		std::size_t operator()(const GraphicsPipelineKey& key) const {
			std::size_t hash = 0;
			PipelineKeyHash::MixRendering(hash, key.rendering);
			for (const auto id: key.vertex_shader_ids) {
				PipelineKeyHash::Mix(hash, id);
			}
			PipelineKeyHash::Mix(hash, key.ps_shader_id);
			PipelineKeyHash::Mix(hash, key.vertex_input.binding_count);
			for (uint32_t i = 0; i < key.vertex_input.binding_count; i++) {
				PipelineKeyHash::Mix(hash, key.vertex_input.bindings[i].stride);
				PipelineKeyHash::Mix(hash, key.vertex_input.bindings[i].instance);
			}
			PipelineKeyHash::Mix(hash, key.vertex_input.attribute_count);
			for (uint32_t i = 0; i < key.vertex_input.attribute_count; i++) {
				PipelineKeyHash::Mix(hash, key.vertex_input.attributes[i].offset);
				PipelineKeyHash::Mix(hash, key.vertex_input.attributes[i].binding);
			}
			PipelineKeyHash::MixStaticParams(hash, key.static_params);
			return hash;
		}
	};

	GraphicContext&               m_graphics;
	std::unique_ptr<ProgramCache> m_program_cache;
	vk::PipelineCache             m_driver_cache = nullptr;
	std::filesystem::path         m_driver_cache_path;
	std::unique_ptr<PipelineLibraryCache> m_libraries;
	uint64_t                              m_graphics_pipelines_created = 0;
	uint64_t                              m_compute_pipelines_created  = 0;
	// Prefetched compute pipelines by program id: layouts made, pipeline compiling in the
	// library cache under ComputePrefetchKey.
	std::unordered_map<uint64_t, std::unique_ptr<Pipeline>> m_compute_prefetched;
	std::unordered_map<GraphicsPipelineKey, std::unique_ptr<Pipeline>, GraphicsPipelineKeyHash>
	                                                        m_graphics_pipelines;
	// The last pipeline GetGraphicsPipeline returned and its key, under m_mutex: consecutive draws
	// mostly use the same pipeline, and comparing keys is cheaper than hashing one.
	GraphicsPipelineKey m_last_graphics_key;
	Pipeline*           m_last_graphics_pipeline = nullptr;
	// Asynchronous pipelines: draws skipped so far for each pipeline whose parts are compiling.
	std::unordered_map<GraphicsPipelineKey, uint32_t, GraphicsPipelineKeyHash> m_deferred_draws;
	std::unordered_map<uint64_t, std::unique_ptr<Pipeline>> m_compute_pipelines;
	Common::Mutex m_mutex;
	std::jthread                                            m_precompile_thread;
	std::mutex                                              m_precompile_join_mutex;
	std::atomic_bool                                        m_precompile_done {true};

	void InitializeDriverCache();
	void InitializeShaderPrecompile();
	void InstallOptimizedPipeline(Pipeline& pipeline, CommandBuffer& command);
	void WaitForPrecompile();
	void ReplayPrecompiled(std::vector<ShaderPrecompile::PermutationRecord> records);
};

void LogPipelineTrace(const char* phase, uint64_t vertex_program_id, uint64_t pixel_program_id);
// Creates a graphics pipeline, from pipeline-library parts when `libraries` is given and the
// pipeline qualifies. Returns -1 for a monolithic pipeline, otherwise a bit per library part it
// compiled (vertex input, pre-rasterization, fragment shader, fragment output).
int CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                           const PipelineRenderingState&          rendering,
                           const PipelineVertexInputState&        vertex_input,
                           std::span<const ShaderVertexInputInfo> vertex_info,
                           const ShaderPixelInputInfo*            ps_input_info,
                           const PipelineCache::GraphicsPrograms& programs,
                           const PipelineStaticParameters&        static_params,
                           PipelineLibraryCache* libraries, vk::PipelineCache driver_cache);
// Compiles the missing shader library parts of a graphics pipeline on the library cache's worker
// threads. Returns the number of parts queued. `ready`, when given, receives whether the
// pipeline can be created without waiting for a shader part to compile.
uint32_t PrefetchLibraryParts(GraphicContext& graphics, const PipelineRenderingState& rendering,
                              const PipelineVertexInputState&        vertex_input,
                              std::span<const ShaderVertexInputInfo> vertex_info,
                              const ShaderPixelInputInfo*            ps_input_info,
                              const PipelineCache::GraphicsPrograms& programs,
                              const PipelineStaticParameters&        static_params,
                              PipelineLibraryCache& libraries, vk::PipelineCache driver_cache,
                              bool* ready = nullptr);
// Creates a compute pipeline's layouts and returns the call that creates the pipeline itself. That
// call reads only its own copies, so it may run on another thread; it returns null on failure.
Common::UniqueFunction<vk::Pipeline> PrepareComputePipeline(GraphicContext&               graphics,
                                                            PipelineCache::Pipeline&      pipeline,
                                                            const ShaderComputeInputInfo& input_info,
                                                            vk::ShaderModule  compute_module,
                                                            vk::PipelineCache driver_cache);
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const ShaderComputeInputInfo& input_info,
                            vk::ShaderModule compute_module, vk::PipelineCache driver_cache);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
