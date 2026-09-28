#include "graphics/host_gpu/renderer/commandHooks.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/profiler.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <intrin.h>
#include <thread>
#include <type_traits>
#include <vector>

namespace Libs::Graphics {

namespace {

// Every Vulkan function the renderer records commands with, plus descriptor updates, command
// buffer begin/end and queue submits.
#define KYTY_HOOKED_VK_FUNCTIONS(X)                                                               \
	X(vkBeginCommandBuffer)                                                                       \
	X(vkEndCommandBuffer)                                                                         \
	X(vkUpdateDescriptorSets)                                                                     \
	X(vkQueueSubmit)                                                                              \
	X(vkCmdPipelineBarrier)                                                                       \
	X(vkCmdPipelineBarrier2)                                                                      \
	X(vkCmdBindPipeline)                                                                          \
	X(vkCmdBindDescriptorSets)                                                                    \
	X(vkCmdPushDescriptorSetKHR)                                                                  \
	X(vkCmdPushConstants)                                                                         \
	X(vkCmdBindVertexBuffers2)                                                                    \
	X(vkCmdBindIndexBuffer)                                                                       \
	X(vkCmdBeginRendering)                                                                        \
	X(vkCmdEndRendering)                                                                          \
	X(vkCmdDraw)                                                                                  \
	X(vkCmdDrawIndexed)                                                                           \
	X(vkCmdDrawIndexedIndirect)                                                                   \
	X(vkCmdDrawMeshTasksEXT)                                                                      \
	X(vkCmdDrawMeshTasksIndirectEXT)                                                              \
	X(vkCmdDispatch)                                                                              \
	X(vkCmdDispatchIndirect)                                                                      \
	X(vkCmdCopyBuffer)                                                                            \
	X(vkCmdCopyImage)                                                                             \
	X(vkCmdCopyBufferToImage)                                                                     \
	X(vkCmdCopyImageToBuffer)                                                                     \
	X(vkCmdResolveImage)                                                                          \
	X(vkCmdFillBuffer)                                                                            \
	X(vkCmdClearColorImage)                                                                       \
	X(vkCmdClearDepthStencilImage)                                                                \
	X(vkCmdClearAttachments)                                                                      \
	X(vkCmdWriteTimestamp)                                                                        \
	X(vkCmdResetQueryPool)                                                                        \
	X(vkCmdBeginConditionalRenderingEXT)                                                          \
	X(vkCmdEndConditionalRenderingEXT)                                                            \
	X(vkCmdSetViewport)                                                                           \
	X(vkCmdSetViewportWithCount)                                                                  \
	X(vkCmdSetScissor)                                                                            \
	X(vkCmdSetScissorWithCount)                                                                   \
	X(vkCmdSetLineWidth)                                                                          \
	X(vkCmdSetDepthBias)                                                                          \
	X(vkCmdSetDepthBiasEnable)                                                                    \
	X(vkCmdSetBlendConstants)                                                                     \
	X(vkCmdSetDepthTestEnable)                                                                    \
	X(vkCmdSetDepthWriteEnable)                                                                   \
	X(vkCmdSetDepthCompareOp)                                                                     \
	X(vkCmdSetStencilTestEnable)                                                                  \
	X(vkCmdSetStencilOp)                                                                          \
	X(vkCmdSetStencilCompareMask)                                                                 \
	X(vkCmdSetStencilWriteMask)                                                                   \
	X(vkCmdSetStencilReference)                                                                   \
	X(vkCmdSetColorWriteEnableEXT)                                                                \
	X(vkCmdSetAttachmentFeedbackLoopEnableEXT)

// Pool calls: with recording deferred, every stream runs what it holds before them, so only one
// thread uses a scheduler's command pool at a time.
#define KYTY_DRAINING_VK_FUNCTIONS(X)                                                             \
	X(vkAllocateCommandBuffers)                                                                   \
	X(vkFreeCommandBuffers)                                                                       \
	X(vkResetCommandPool)                                                                         \
	X(vkDestroyCommandPool)

// Every other command (and command buffer reset): into a routed buffer, it runs after what the
// stream holds, recorded directly. None is common enough to be worth copying.
#define KYTY_FALLBACK_VK_FUNCTIONS(X)                                                             \
	X(vkCmdUpdateBuffer)                                                                          \
	X(vkCmdBeginQuery)                                                                            \
	X(vkCmdEndQuery)                                                                              \
	X(vkCmdCopyQueryPoolResults)                                                                  \
	X(vkCmdExecuteCommands)                                                                       \
	X(vkCmdSetEvent)                                                                              \
	X(vkCmdResetEvent)                                                                            \
	X(vkCmdWaitEvents)                                                                            \
	X(vkCmdSetDepthBounds)                                                                        \
	X(vkCmdBindVertexBuffers)                                                                     \
	X(vkCmdDrawIndirect)                                                                          \
	X(vkCmdBlitImage)                                                                             \
	X(vkCmdBeginRenderPass)                                                                       \
	X(vkCmdNextSubpass)                                                                           \
	X(vkCmdEndRenderPass)                                                                         \
	X(vkCmdSetDeviceMask)                                                                         \
	X(vkCmdDispatchBase)                                                                          \
	X(vkCmdDrawIndirectCount)                                                                     \
	X(vkCmdDrawIndexedIndirectCount)                                                              \
	X(vkCmdBeginRenderPass2)                                                                      \
	X(vkCmdNextSubpass2)                                                                          \
	X(vkCmdEndRenderPass2)                                                                        \
	X(vkCmdWriteTimestamp2)                                                                       \
	X(vkCmdCopyBuffer2)                                                                           \
	X(vkCmdCopyImage2)                                                                            \
	X(vkCmdCopyBufferToImage2)                                                                    \
	X(vkCmdCopyImageToBuffer2)                                                                    \
	X(vkCmdSetEvent2)                                                                             \
	X(vkCmdResetEvent2)                                                                           \
	X(vkCmdWaitEvents2)                                                                           \
	X(vkCmdBlitImage2)                                                                            \
	X(vkCmdResolveImage2)                                                                         \
	X(vkCmdSetCullMode)                                                                           \
	X(vkCmdSetFrontFace)                                                                          \
	X(vkCmdSetPrimitiveTopology)                                                                  \
	X(vkCmdSetDepthBoundsTestEnable)                                                              \
	X(vkCmdSetRasterizerDiscardEnable)                                                            \
	X(vkCmdSetPrimitiveRestartEnable)                                                             \
	X(vkCmdPushDescriptorSet)                                                                     \
	X(vkCmdPushDescriptorSetWithTemplate)                                                         \
	X(vkCmdBindDescriptorSets2)                                                                   \
	X(vkCmdPushConstants2)                                                                        \
	X(vkCmdPushDescriptorSet2)                                                                    \
	X(vkCmdPushDescriptorSetWithTemplate2)                                                        \
	X(vkCmdSetLineStipple)                                                                        \
	X(vkCmdBindIndexBuffer2)                                                                      \
	X(vkCmdSetRenderingAttachmentLocations)                                                       \
	X(vkCmdSetRenderingInputAttachmentIndices)                                                    \
	X(vkCmdDebugMarkerBeginEXT)                                                                   \
	X(vkCmdDebugMarkerEndEXT)                                                                     \
	X(vkCmdDebugMarkerInsertEXT)                                                                  \
	X(vkCmdBeginVideoCodingKHR)                                                                   \
	X(vkCmdEndVideoCodingKHR)                                                                     \
	X(vkCmdControlVideoCodingKHR)                                                                 \
	X(vkCmdDecodeVideoKHR)                                                                        \
	X(vkCmdBindTransformFeedbackBuffersEXT)                                                       \
	X(vkCmdBeginTransformFeedbackEXT)                                                             \
	X(vkCmdEndTransformFeedbackEXT)                                                               \
	X(vkCmdBeginQueryIndexedEXT)                                                                  \
	X(vkCmdEndQueryIndexedEXT)                                                                    \
	X(vkCmdDrawIndirectByteCountEXT)                                                              \
	X(vkCmdCuLaunchKernelNVX)                                                                     \
	X(vkCmdDrawIndirectCountAMD)                                                                  \
	X(vkCmdDrawIndexedIndirectCountAMD)                                                           \
	X(vkCmdBeginRenderingKHR)                                                                     \
	X(vkCmdEndRenderingKHR)                                                                       \
	X(vkCmdSetDeviceMaskKHR)                                                                      \
	X(vkCmdDispatchBaseKHR)                                                                       \
	X(vkCmdPushDescriptorSetWithTemplateKHR)                                                      \
	X(vkCmdSetViewportWScalingNV)                                                                 \
	X(vkCmdSetDiscardRectangleEXT)                                                                \
	X(vkCmdSetDiscardRectangleEnableEXT)                                                          \
	X(vkCmdSetDiscardRectangleModeEXT)                                                            \
	X(vkCmdBeginRenderPass2KHR)                                                                   \
	X(vkCmdNextSubpass2KHR)                                                                       \
	X(vkCmdEndRenderPass2KHR)                                                                     \
	X(vkCmdBeginDebugUtilsLabelEXT)                                                               \
	X(vkCmdEndDebugUtilsLabelEXT)                                                                 \
	X(vkCmdInsertDebugUtilsLabelEXT)                                                              \
	X(vkCmdSetSampleLocationsEXT)                                                                 \
	X(vkCmdBuildAccelerationStructuresKHR)                                                        \
	X(vkCmdBuildAccelerationStructuresIndirectKHR)                                                \
	X(vkCmdCopyAccelerationStructureKHR)                                                          \
	X(vkCmdCopyAccelerationStructureToMemoryKHR)                                                  \
	X(vkCmdCopyMemoryToAccelerationStructureKHR)                                                  \
	X(vkCmdWriteAccelerationStructuresPropertiesKHR)                                              \
	X(vkCmdTraceRaysKHR)                                                                          \
	X(vkCmdTraceRaysIndirectKHR)                                                                  \
	X(vkCmdSetRayTracingPipelineStackSizeKHR)                                                     \
	X(vkCmdBindShadingRateImageNV)                                                                \
	X(vkCmdSetViewportShadingRatePaletteNV)                                                       \
	X(vkCmdSetCoarseSampleOrderNV)                                                                \
	X(vkCmdBuildAccelerationStructureNV)                                                          \
	X(vkCmdCopyAccelerationStructureNV)                                                           \
	X(vkCmdTraceRaysNV)                                                                           \
	X(vkCmdWriteAccelerationStructuresPropertiesNV)                                               \
	X(vkCmdDrawIndirectCountKHR)                                                                  \
	X(vkCmdDrawIndexedIndirectCountKHR)                                                           \
	X(vkCmdWriteBufferMarkerAMD)                                                                  \
	X(vkCmdWriteBufferMarker2AMD)                                                                 \
	X(vkCmdDrawMeshTasksNV)                                                                       \
	X(vkCmdDrawMeshTasksIndirectNV)                                                               \
	X(vkCmdDrawMeshTasksIndirectCountNV)                                                          \
	X(vkCmdSetExclusiveScissorEnableNV)                                                           \
	X(vkCmdSetExclusiveScissorNV)                                                                 \
	X(vkCmdSetCheckpointNV)                                                                       \
	X(vkCmdSetPerformanceMarkerINTEL)                                                             \
	X(vkCmdSetPerformanceStreamMarkerINTEL)                                                       \
	X(vkCmdSetPerformanceOverrideINTEL)                                                           \
	X(vkCmdSetFragmentShadingRateKHR)                                                             \
	X(vkCmdSetRenderingAttachmentLocationsKHR)                                                    \
	X(vkCmdSetRenderingInputAttachmentIndicesKHR)                                                 \
	X(vkCmdSetLineStippleEXT)                                                                     \
	X(vkCmdSetCullModeEXT)                                                                        \
	X(vkCmdSetFrontFaceEXT)                                                                       \
	X(vkCmdSetPrimitiveTopologyEXT)                                                               \
	X(vkCmdSetViewportWithCountEXT)                                                               \
	X(vkCmdSetScissorWithCountEXT)                                                                \
	X(vkCmdBindVertexBuffers2EXT)                                                                 \
	X(vkCmdSetDepthTestEnableEXT)                                                                 \
	X(vkCmdSetDepthWriteEnableEXT)                                                                \
	X(vkCmdSetDepthCompareOpEXT)                                                                  \
	X(vkCmdSetDepthBoundsTestEnableEXT)                                                           \
	X(vkCmdSetStencilTestEnableEXT)                                                               \
	X(vkCmdSetStencilOpEXT)                                                                       \
	X(vkCmdPreprocessGeneratedCommandsNV)                                                         \
	X(vkCmdExecuteGeneratedCommandsNV)                                                            \
	X(vkCmdBindPipelineShaderGroupNV)                                                             \
	X(vkCmdSetDepthBias2EXT)                                                                      \
	X(vkCmdEncodeVideoKHR)                                                                        \
	X(vkCmdDispatchTileQCOM)                                                                      \
	X(vkCmdBeginPerTileExecutionQCOM)                                                             \
	X(vkCmdEndPerTileExecutionQCOM)                                                               \
	X(vkCmdSetEvent2KHR)                                                                          \
	X(vkCmdResetEvent2KHR)                                                                        \
	X(vkCmdWaitEvents2KHR)                                                                        \
	X(vkCmdPipelineBarrier2KHR)                                                                   \
	X(vkCmdWriteTimestamp2KHR)                                                                    \
	X(vkCmdBindDescriptorBuffersEXT)                                                              \
	X(vkCmdSetDescriptorBufferOffsetsEXT)                                                         \
	X(vkCmdBindDescriptorBufferEmbeddedSamplersEXT)                                               \
	X(vkCmdSetFragmentShadingRateEnumNV)                                                          \
	X(vkCmdDrawMeshTasksIndirectCountEXT)                                                         \
	X(vkCmdCopyBuffer2KHR)                                                                        \
	X(vkCmdCopyImage2KHR)                                                                         \
	X(vkCmdCopyBufferToImage2KHR)                                                                 \
	X(vkCmdCopyImageToBuffer2KHR)                                                                 \
	X(vkCmdBlitImage2KHR)                                                                         \
	X(vkCmdResolveImage2KHR)                                                                      \
	X(vkCmdSetVertexInputEXT)                                                                     \
	X(vkCmdSubpassShadingHUAWEI)                                                                  \
	X(vkCmdBindInvocationMaskHUAWEI)                                                              \
	X(vkCmdSetPatchControlPointsEXT)                                                              \
	X(vkCmdSetRasterizerDiscardEnableEXT)                                                         \
	X(vkCmdSetDepthBiasEnableEXT)                                                                 \
	X(vkCmdSetLogicOpEXT)                                                                         \
	X(vkCmdSetPrimitiveRestartEnableEXT)                                                          \
	X(vkCmdTraceRaysIndirect2KHR)                                                                 \
	X(vkCmdDrawMultiEXT)                                                                          \
	X(vkCmdDrawMultiIndexedEXT)                                                                   \
	X(vkCmdBuildMicromapsEXT)                                                                     \
	X(vkCmdCopyMicromapEXT)                                                                       \
	X(vkCmdCopyMicromapToMemoryEXT)                                                               \
	X(vkCmdCopyMemoryToMicromapEXT)                                                               \
	X(vkCmdWriteMicromapsPropertiesEXT)                                                           \
	X(vkCmdDrawClusterHUAWEI)                                                                     \
	X(vkCmdDrawClusterIndirectHUAWEI)                                                             \
	X(vkCmdCopyMemoryIndirectNV)                                                                  \
	X(vkCmdCopyMemoryToImageIndirectNV)                                                           \
	X(vkCmdDecompressMemoryNV)                                                                    \
	X(vkCmdDecompressMemoryIndirectCountNV)                                                       \
	X(vkCmdUpdatePipelineIndirectBufferNV)                                                        \
	X(vkCmdSetDepthClampEnableEXT)                                                                \
	X(vkCmdSetPolygonModeEXT)                                                                     \
	X(vkCmdSetRasterizationSamplesEXT)                                                            \
	X(vkCmdSetSampleMaskEXT)                                                                      \
	X(vkCmdSetAlphaToCoverageEnableEXT)                                                           \
	X(vkCmdSetAlphaToOneEnableEXT)                                                                \
	X(vkCmdSetLogicOpEnableEXT)                                                                   \
	X(vkCmdSetColorBlendEnableEXT)                                                                \
	X(vkCmdSetColorBlendEquationEXT)                                                              \
	X(vkCmdSetColorWriteMaskEXT)                                                                  \
	X(vkCmdSetTessellationDomainOriginEXT)                                                        \
	X(vkCmdSetRasterizationStreamEXT)                                                             \
	X(vkCmdSetConservativeRasterizationModeEXT)                                                   \
	X(vkCmdSetExtraPrimitiveOverestimationSizeEXT)                                                \
	X(vkCmdSetDepthClipEnableEXT)                                                                 \
	X(vkCmdSetSampleLocationsEnableEXT)                                                           \
	X(vkCmdSetColorBlendAdvancedEXT)                                                              \
	X(vkCmdSetProvokingVertexModeEXT)                                                             \
	X(vkCmdSetLineRasterizationModeEXT)                                                           \
	X(vkCmdSetLineStippleEnableEXT)                                                               \
	X(vkCmdSetDepthClipNegativeOneToOneEXT)                                                       \
	X(vkCmdSetViewportWScalingEnableNV)                                                           \
	X(vkCmdSetViewportSwizzleNV)                                                                  \
	X(vkCmdSetCoverageToColorEnableNV)                                                            \
	X(vkCmdSetCoverageToColorLocationNV)                                                          \
	X(vkCmdSetCoverageModulationModeNV)                                                           \
	X(vkCmdSetCoverageModulationTableEnableNV)                                                    \
	X(vkCmdSetCoverageModulationTableNV)                                                          \
	X(vkCmdSetShadingRateImageEnableNV)                                                           \
	X(vkCmdSetRepresentativeFragmentTestEnableNV)                                                 \
	X(vkCmdSetCoverageReductionModeNV)                                                            \
	X(vkCmdCopyTensorARM)                                                                         \
	X(vkCmdOpticalFlowExecuteNV)                                                                  \
	X(vkCmdBindIndexBuffer2KHR)                                                                   \
	X(vkCmdBindShadersEXT)                                                                        \
	X(vkCmdSetDepthClampRangeEXT)                                                                 \
	X(vkCmdConvertCooperativeVectorMatrixNV)                                                      \
	X(vkCmdDispatchDataGraphARM)                                                                  \
	X(vkCmdSetLineStippleKHR)                                                                     \
	X(vkCmdBindDescriptorSets2KHR)                                                                \
	X(vkCmdPushConstants2KHR)                                                                     \
	X(vkCmdPushDescriptorSet2KHR)                                                                 \
	X(vkCmdPushDescriptorSetWithTemplate2KHR)                                                     \
	X(vkCmdSetDescriptorBufferOffsets2EXT)                                                        \
	X(vkCmdBindDescriptorBufferEmbeddedSamplers2EXT)                                              \
	X(vkCmdBindTileMemoryQCOM)                                                                    \
	X(vkCmdCopyMemoryIndirectKHR)                                                                 \
	X(vkCmdCopyMemoryToImageIndirectKHR)                                                          \
	X(vkCmdDecompressMemoryEXT)                                                                   \
	X(vkCmdDecompressMemoryIndirectCountEXT)                                                      \
	X(vkCmdBuildClusterAccelerationStructureIndirectNV)                                           \
	X(vkCmdBuildPartitionedAccelerationStructuresNV)                                              \
	X(vkCmdPreprocessGeneratedCommandsEXT)                                                        \
	X(vkCmdExecuteGeneratedCommandsEXT)                                                           \
	X(vkCmdEndRendering2EXT)                                                                      \
	X(vkCmdBeginCustomResolveEXT)                                                                 \
	X(vkCmdEndRendering2KHR)                                                                      \
	X(vkResetCommandBuffer)

// ---------------------------------------------------------------------------------------------
// Timing (KYTY_DEBUG_VK_TIME=1)

thread_local bool     t_timed_thread = false;
std::atomic<uint64_t> g_ticks {0};
std::atomic<uint64_t> g_calls {0};

void Report() {
	using Clock = std::chrono::steady_clock;
	thread_local auto window_start = Clock::now();
	thread_local auto tsc_start    = __rdtsc();
	const auto        now          = Clock::now();
	if (now - window_start < std::chrono::seconds(5)) {
		return;
	}
	const auto seconds = std::chrono::duration<double>(now - window_start).count();
	const auto tsc     = __rdtsc();
	const auto ticks   = g_ticks.exchange(0, std::memory_order_relaxed);
	const auto calls   = g_calls.exchange(0, std::memory_order_relaxed);
	// The share of the GPU thread's wall time spent inside the hooked calls.
	const auto share = static_cast<double>(ticks) / static_cast<double>(tsc - tsc_start);
	std::printf("vk-time: %.1fs calls/s=%.0f ms/s=%.1f\n", seconds,
	            static_cast<double>(calls) / seconds, share * 1000.0);
	window_start = now;
	tsc_start    = tsc;
}

template <typename Tag, typename Fn>
struct TimedHook;

template <typename Tag, typename R, typename... A>
struct TimedHook<Tag, R(VKAPI_PTR*)(A...)> {
	static inline R(VKAPI_PTR* real)(A...) = nullptr;

	static R VKAPI_PTR Call(A... args) {
		if (!t_timed_thread) {
			return real(args...);
		}
		const auto start = __rdtsc();
		if constexpr (std::is_void_v<R>) {
			real(args...);
			g_ticks.fetch_add(__rdtsc() - start, std::memory_order_relaxed);
			g_calls.fetch_add(1, std::memory_order_relaxed);
			Report();
		} else {
			R result = real(args...);
			g_ticks.fetch_add(__rdtsc() - start, std::memory_order_relaxed);
			g_calls.fetch_add(1, std::memory_order_relaxed);
			Report();
			return result;
		}
	}
};

// ---------------------------------------------------------------------------------------------
// Deferred recording (the "record-thread" setting; KYTY_RECORD_THREAD=0/1 overrides it)

struct RealFunctions {
#define KYTY_DECLARE_REAL(name) PFN_##name name = nullptr;
	KYTY_HOOKED_VK_FUNCTIONS(KYTY_DECLARE_REAL)
	KYTY_DRAINING_VK_FUNCTIONS(KYTY_DECLARE_REAL)
#undef KYTY_DECLARE_REAL
};
RealFunctions g_real;
bool          g_deferred = false;

} // namespace

// Packets: a header, copies of the arrays the call points to, then the call's closure.
struct CommandStream::Impl {
	struct Header {
		void (*run)(void* closure) = nullptr; // Null: skip to the ring's start.
		uint32_t size              = 0;       // The whole packet.
		uint32_t closure_offset    = 0;       // From the header; the closure follows the arrays.
	};
	static constexpr size_t RingBytes    = size_t {64} << 20u;
	static constexpr size_t Align        = 16;
	static constexpr size_t ClosureBytes = 176;
	static constexpr size_t HeaderBytes  = (sizeof(Header) + Align - 1) & ~(Align - 1);
	// A consumer asleep is woken after this much more work (or at a submit or drain).
	static constexpr uint64_t WakeBytes = 16u << 10u;

	std::vector<uint8_t>  ring = std::vector<uint8_t>(RingBytes);
	std::atomic<uint64_t> write {0}; // Bytes appended (published).
	std::atomic<uint64_t> read {0};  // Consumer: bytes run.
	std::atomic<uint64_t> signal {0};
	std::atomic<bool>     sleeping {false};
	// Held by the thread appending a packet. Recording into a command buffer is externally
	// synchronized already; this also orders descriptor updates from the routing thread.
	std::atomic<bool>     appending {false};
	uint64_t              pending_write = 0; // Appending thread: bytes reserved, not yet published.
	uint64_t              wake_mark     = 0;

	void BeginAppend() {
		while (appending.exchange(true, std::memory_order_acquire)) {
			while (appending.load(std::memory_order_relaxed)) {
				_mm_pause();
			}
		}
	}
	void EndAppend() {
		appending.store(false, std::memory_order_release);
	}

	// Reserves a packet of `bytes` (header and closure included) and returns its start.
	uint8_t* Reserve(size_t bytes) {
		bytes = (bytes + Align - 1) & ~(Align - 1);
		EXIT_IF(bytes > RingBytes / 4u);
		auto offset = pending_write % RingBytes;
		if (RingBytes - offset < bytes) {
			// The packet would cross the end: a skip header takes the rest of the ring.
			WaitForSpace(RingBytes - offset);
			auto* header = reinterpret_cast<Header*>(ring.data() + offset);
			*header      = {};
			pending_write += RingBytes - offset;
			write.store(pending_write, std::memory_order_release);
			offset = 0;
		}
		WaitForSpace(bytes);
		return ring.data() + offset;
	}
	void WaitForSpace(size_t bytes) {
		while (pending_write + bytes - read.load(std::memory_order_acquire) > RingBytes) {
			if (sleeping.load(std::memory_order_seq_cst)) {
				Signal();
			}
			std::this_thread::yield();
		}
	}
	void Publish(uint8_t* start, size_t bytes) {
		bytes = (bytes + Align - 1) & ~(Align - 1);
		EXIT_IF(start != ring.data() + pending_write % RingBytes);
		pending_write += bytes;
		packets.store(packets.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
		// Sequentially consistent with the consumer's "sleeping, then check write", so a
		// consumer about to sleep either sees this packet or is seen asleep.
		write.store(pending_write, std::memory_order_seq_cst);
		if (pending_write - wake_mark >= WakeBytes && sleeping.load(std::memory_order_seq_cst)) {
			wake_mark = pending_write;
			Signal();
		}
	}
	void Signal() {
		wakes.fetch_add(1, std::memory_order_relaxed);
		signal.fetch_add(1, std::memory_order_release);
		signal.notify_one();
	}
	// Any thread: waits until the consumer has run everything published so far.
	void Drain();
	// Statistics (KYTY_DEBUG_STREAM_STATS=1).
	std::atomic<uint64_t> packets {0};
	std::atomic<uint64_t> wakes {0};
	std::atomic<uint64_t> drains {0};
	std::atomic<uint64_t> sleeps {0};
};

// Streams by the command buffer each routes. A slot's stream is created once and kept for the
// process, so any thread may look one up while a scheduler goes away.
struct CommandStream::Slot {
	std::atomic<bool>                 in_use {false};
	std::atomic<VkCommandBuffer>      buffer {VK_NULL_HANDLE};
	std::atomic<CommandStream::Impl*> stream {nullptr};
};

namespace {

std::array<CommandStream::Slot, 4> g_slots;
std::atomic<uint64_t>              g_fallback_calls {0};

// The stream the current routed call appends to.
thread_local CommandStream::Impl* t_stream = nullptr;
// The stream this thread routes buffers through (it begins and submits them), if any.
thread_local CommandStream::Slot* t_home = nullptr;
// The stream this thread runs, if it is a consumer.
thread_local CommandStream::Impl* t_consuming = nullptr;

// The stream `buffer` records through, if it is routed. The thread recording into a buffer owns
// it (Vulkan requires external synchronization), so it sees the routing ordered before that.
[[nodiscard]] CommandStream::Impl* StreamFor(VkCommandBuffer buffer) {
	if (buffer != VK_NULL_HANDLE) {
		for (auto& slot: g_slots) {
			if (slot.buffer.load(std::memory_order_relaxed) == buffer) {
				return slot.stream.load(std::memory_order_relaxed);
			}
		}
	}
	return nullptr;
}

[[nodiscard]] bool Routed(VkCommandBuffer buffer) {
	t_stream = StreamFor(buffer);
	return t_stream != nullptr;
}

// A direct call on a thread that routes a stream: run everything queued first.
void BeforeDirect() {
	if (t_home != nullptr) {
		t_home->stream.load(std::memory_order_acquire)->Drain();
	}
}

// Pool calls: every stream runs what it holds first, since its consumer may be recording into a
// buffer from the pool.
void DrainAll() {
	for (auto& slot: g_slots) {
		if (auto* stream = slot.stream.load(std::memory_order_acquire); stream != nullptr) {
			stream->Drain();
		}
	}
}

} // namespace

void CommandStream::Impl::Drain() {
	// A consumer has already run everything before what it is running now.
	const auto target = write.load(std::memory_order_seq_cst);
	if (t_consuming == this || read.load(std::memory_order_acquire) >= target) {
		return;
	}
	drains.fetch_add(1, std::memory_order_relaxed);
	if (sleeping.load(std::memory_order_seq_cst)) {
		Signal();
	}
	// The consumer takes a while to wake: waking it again on every spin would cost a kernel
	// call each. Re-wake only now and then, for a consumer that went back to sleep.
	for (uint32_t spins = 1; read.load(std::memory_order_acquire) < target; spins++) {
		_mm_pause();
		if ((spins & 0xfffu) == 0 && sleeping.load(std::memory_order_seq_cst)) {
			Signal();
		}
	}
}

namespace {


template <typename T>
[[nodiscard]] constexpr size_t Bytes(uint32_t count) {
	return (sizeof(T) * count + CommandStream::Impl::Align - 1) & ~(CommandStream::Impl::Align - 1);
}

// Builds one packet: copies of arrays, then the closure that makes the call.
class Packet {
public:
	// Reserves room for the arrays and the largest closure; publishes only what it uses.
	Packet(CommandStream::Impl& stream, size_t array_bytes)
	    : m_stream(stream), m_size(CommandStream::Impl::HeaderBytes + array_bytes +
	                               CommandStream::Impl::ClosureBytes),
	      m_start(Start(stream, m_size)), m_next(m_start + CommandStream::Impl::HeaderBytes) {}
	~Packet() {
		m_stream.EndAppend();
	}
	KYTY_CLASS_NO_COPY(Packet);

	template <typename T>
	[[nodiscard]] T* Copy(const T* source, uint32_t count) {
		if (source == nullptr || count == 0) {
			return nullptr;
		}
		auto* target = reinterpret_cast<T*>(m_next);
		std::memcpy(target, source, sizeof(T) * count);
		m_next += Bytes<T>(count);
		EXIT_IF(m_next + CommandStream::Impl::ClosureBytes > m_start + m_size);
		return target;
	}

	template <typename Call>
	void Commit(Call call) {
		static_assert(std::is_trivially_destructible_v<Call>);
		static_assert(sizeof(Call) <= CommandStream::Impl::ClosureBytes);
		auto* header = reinterpret_cast<CommandStream::Impl::Header*>(m_start);
		::new (m_next) Call(call);
		header->run            = [](void* closure) { (*static_cast<Call*>(closure))(); };
		header->closure_offset = static_cast<uint32_t>(m_next - m_start);
		const auto used        = static_cast<size_t>(m_next - m_start) + sizeof(Call);
		header->size = static_cast<uint32_t>((used + CommandStream::Impl::Align - 1) &
		                                     ~(CommandStream::Impl::Align - 1));
		m_stream.Publish(m_start, header->size);
	}

private:
	static uint8_t* Start(CommandStream::Impl& stream, size_t bytes) {
		stream.BeginAppend();
		return stream.Reserve(bytes);
	}

	CommandStream::Impl& m_stream;
	size_t               m_size;
	uint8_t*             m_start;
	uint8_t*             m_next;
};

template <typename Call>
void Emit(Call call) {
	Packet(*t_stream, 0).Commit(call);
}

void RequireNoNext(const void* next, const char* what) {
	if (next != nullptr) {
		EXIT("deferred recording: %s with a pNext chain is not supported\n", what);
	}
}

// Which array a descriptor write reads, by its type (Vulkan ignores, and allows garbage in, the
// other two pointers).
enum class WriteArray { Image, Buffer, TexelBuffer };

[[nodiscard]] WriteArray WriteArrayOf(VkDescriptorType type) {
	switch (type) {
		case VK_DESCRIPTOR_TYPE_SAMPLER:
		case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
		case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
		case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
		case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT: return WriteArray::Image;
		case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
		case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER: return WriteArray::TexelBuffer;
		case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
		case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
		case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
		case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC: return WriteArray::Buffer;
		default:
			EXIT("deferred recording: descriptor type %d is not supported\n", static_cast<int>(type));
	}
}

[[nodiscard]] size_t WritesBytes(uint32_t count, const VkWriteDescriptorSet* writes) {
	size_t bytes = Bytes<VkWriteDescriptorSet>(count);
	for (uint32_t i = 0; i < count; i++) {
		switch (WriteArrayOf(writes[i].descriptorType)) {
			case WriteArray::Image: bytes += Bytes<VkDescriptorImageInfo>(writes[i].descriptorCount); break;
			case WriteArray::Buffer: bytes += Bytes<VkDescriptorBufferInfo>(writes[i].descriptorCount); break;
			case WriteArray::TexelBuffer: bytes += Bytes<VkBufferView>(writes[i].descriptorCount); break;
		}
	}
	return bytes;
}

[[nodiscard]] VkWriteDescriptorSet* CopyWrites(Packet& packet, uint32_t count,
                                               const VkWriteDescriptorSet* writes) {
	auto* copies = packet.Copy(writes, count);
	for (uint32_t i = 0; i < count; i++) {
		RequireNoNext(writes[i].pNext, "a descriptor write");
		copies[i].pImageInfo       = nullptr;
		copies[i].pBufferInfo      = nullptr;
		copies[i].pTexelBufferView = nullptr;
		switch (WriteArrayOf(writes[i].descriptorType)) {
			case WriteArray::Image:
				copies[i].pImageInfo = packet.Copy(writes[i].pImageInfo, writes[i].descriptorCount);
				break;
			case WriteArray::Buffer:
				copies[i].pBufferInfo = packet.Copy(writes[i].pBufferInfo, writes[i].descriptorCount);
				break;
			case WriteArray::TexelBuffer:
				copies[i].pTexelBufferView =
				    packet.Copy(writes[i].pTexelBufferView, writes[i].descriptorCount);
				break;
		}
	}
	return copies;
}

// --- Hooks: routed calls go into the stream; other calls on a routing thread drain it first.

VkResult VKAPI_PTR HookBeginCommandBuffer(VkCommandBuffer buffer, const VkCommandBufferBeginInfo* info) {
	if (Routed(buffer)) {
		RequireNoNext(info->pNext, "a command buffer begin");
		EXIT_IF(info->pInheritanceInfo != nullptr);
		Packet packet(*t_stream, Bytes<VkCommandBufferBeginInfo>(1));
		const auto* copy = packet.Copy(info, 1);
		packet.Commit([=] {
			EXIT_NOT_IMPLEMENTED(g_real.vkBeginCommandBuffer(buffer, copy) != VK_SUCCESS);
		});
		return VK_SUCCESS;
	}
	BeforeDirect();
	return g_real.vkBeginCommandBuffer(buffer, info);
}

VkResult VKAPI_PTR HookEndCommandBuffer(VkCommandBuffer buffer) {
	if (Routed(buffer)) {
		Emit([=] { EXIT_NOT_IMPLEMENTED(g_real.vkEndCommandBuffer(buffer) != VK_SUCCESS); });
		return VK_SUCCESS;
	}
	BeforeDirect();
	return g_real.vkEndCommandBuffer(buffer);
}

void VKAPI_PTR HookUpdateDescriptorSets(VkDevice device, uint32_t write_count,
                                        const VkWriteDescriptorSet* writes, uint32_t copy_count,
                                        const VkCopyDescriptorSet* copies) {
	if (t_home != nullptr && t_home->buffer.load(std::memory_order_relaxed) != VK_NULL_HANDLE) {
		// While this thread's stream records: in order with the binds recorded around it.
		t_stream = t_home->stream.load(std::memory_order_relaxed);
		Packet packet(*t_stream, WritesBytes(write_count, writes) +
		                             Bytes<VkCopyDescriptorSet>(copy_count));
		const auto* write_copies = CopyWrites(packet, write_count, writes);
		const auto* copy_copies  = packet.Copy(copies, copy_count);
		packet.Commit([=] {
			g_real.vkUpdateDescriptorSets(device, write_count, write_copies, copy_count, copy_copies);
		});
		return;
	}
	// Not recording through the stream: after what it holds, like any direct call.
	BeforeDirect();
	g_real.vkUpdateDescriptorSets(device, write_count, writes, copy_count, copies);
}

VkResult VKAPI_PTR HookQueueSubmit(VkQueue queue, uint32_t count, const VkSubmitInfo* submits,
                                   VkFence fence) {
	// Callers hold the queue lock, which the stream's own submits need: a routing thread drains
	// its stream before taking it (DrainGpuThreadCommands), never here.
	return g_real.vkQueueSubmit(queue, count, submits, fence);
}

void VKAPI_PTR HookCmdPipelineBarrier(VkCommandBuffer buffer, VkPipelineStageFlags src,
                                      VkPipelineStageFlags dst, VkDependencyFlags flags,
                                      uint32_t memory_count, const VkMemoryBarrier* memory,
                                      uint32_t buffer_count, const VkBufferMemoryBarrier* buffers,
                                      uint32_t image_count, const VkImageMemoryBarrier* images) {
	if (Routed(buffer)) {
		Packet      packet(*t_stream, Bytes<VkMemoryBarrier>(memory_count) +
		                                   Bytes<VkBufferMemoryBarrier>(buffer_count) +
		                                   Bytes<VkImageMemoryBarrier>(image_count));
		const auto* m = packet.Copy(memory, memory_count);
		const auto* b = packet.Copy(buffers, buffer_count);
		const auto* i = packet.Copy(images, image_count);
		packet.Commit([=] {
			g_real.vkCmdPipelineBarrier(buffer, src, dst, flags, memory_count, m, buffer_count, b,
			                            image_count, i);
		});
		return;
	}
	BeforeDirect();
	g_real.vkCmdPipelineBarrier(buffer, src, dst, flags, memory_count, memory, buffer_count,
	                            buffers, image_count, images);
}

void VKAPI_PTR HookCmdPipelineBarrier2(VkCommandBuffer buffer, const VkDependencyInfo* info) {
	if (Routed(buffer)) {
		RequireNoNext(info->pNext, "a dependency info");
		Packet packet(*t_stream, Bytes<VkDependencyInfo>(1) +
		                             Bytes<VkMemoryBarrier2>(info->memoryBarrierCount) +
		                             Bytes<VkBufferMemoryBarrier2>(info->bufferMemoryBarrierCount) +
		                             Bytes<VkImageMemoryBarrier2>(info->imageMemoryBarrierCount));
		auto* copy = packet.Copy(info, 1);
		copy->pMemoryBarriers = packet.Copy(info->pMemoryBarriers, info->memoryBarrierCount);
		copy->pBufferMemoryBarriers =
		    packet.Copy(info->pBufferMemoryBarriers, info->bufferMemoryBarrierCount);
		copy->pImageMemoryBarriers =
		    packet.Copy(info->pImageMemoryBarriers, info->imageMemoryBarrierCount);
		packet.Commit([=] { g_real.vkCmdPipelineBarrier2(buffer, copy); });
		return;
	}
	BeforeDirect();
	g_real.vkCmdPipelineBarrier2(buffer, info);
}

void VKAPI_PTR HookCmdBindPipeline(VkCommandBuffer buffer, VkPipelineBindPoint point,
                                   VkPipeline pipeline) {
	if (Routed(buffer)) {
		Emit([=] { g_real.vkCmdBindPipeline(buffer, point, pipeline); });
		return;
	}
	BeforeDirect();
	g_real.vkCmdBindPipeline(buffer, point, pipeline);
}

void VKAPI_PTR HookCmdBindDescriptorSets(VkCommandBuffer buffer, VkPipelineBindPoint point,
                                         VkPipelineLayout layout, uint32_t first, uint32_t count,
                                         const VkDescriptorSet* sets, uint32_t dynamic_count,
                                         const uint32_t* dynamic_offsets) {
	if (Routed(buffer)) {
		Packet      packet(*t_stream,
		                   Bytes<VkDescriptorSet>(count) + Bytes<uint32_t>(dynamic_count));
		const auto* s = packet.Copy(sets, count);
		const auto* d = packet.Copy(dynamic_offsets, dynamic_count);
		packet.Commit([=] {
			g_real.vkCmdBindDescriptorSets(buffer, point, layout, first, count, s, dynamic_count, d);
		});
		return;
	}
	BeforeDirect();
	g_real.vkCmdBindDescriptorSets(buffer, point, layout, first, count, sets, dynamic_count,
	                               dynamic_offsets);
}

void VKAPI_PTR HookCmdPushDescriptorSetKHR(VkCommandBuffer buffer, VkPipelineBindPoint point,
                                           VkPipelineLayout layout, uint32_t set, uint32_t count,
                                           const VkWriteDescriptorSet* writes) {
	if (Routed(buffer)) {
		Packet      packet(*t_stream, WritesBytes(count, writes));
		const auto* copies = CopyWrites(packet, count, writes);
		packet.Commit(
		    [=] { g_real.vkCmdPushDescriptorSetKHR(buffer, point, layout, set, count, copies); });
		return;
	}
	BeforeDirect();
	g_real.vkCmdPushDescriptorSetKHR(buffer, point, layout, set, count, writes);
}

void VKAPI_PTR HookCmdPushConstants(VkCommandBuffer buffer, VkPipelineLayout layout,
                                    VkShaderStageFlags stages, uint32_t offset, uint32_t size,
                                    const void* values) {
	if (Routed(buffer)) {
		Packet      packet(*t_stream, Bytes<uint8_t>(size));
		const auto* copy = packet.Copy(static_cast<const uint8_t*>(values), size);
		packet.Commit([=] { g_real.vkCmdPushConstants(buffer, layout, stages, offset, size, copy); });
		return;
	}
	BeforeDirect();
	g_real.vkCmdPushConstants(buffer, layout, stages, offset, size, values);
}

void VKAPI_PTR HookCmdBindVertexBuffers2(VkCommandBuffer buffer, uint32_t first, uint32_t count,
                                         const VkBuffer* buffers, const VkDeviceSize* offsets,
                                         const VkDeviceSize* sizes, const VkDeviceSize* strides) {
	if (Routed(buffer)) {
		Packet      packet(*t_stream, Bytes<VkBuffer>(count) + 3 * Bytes<VkDeviceSize>(count));
		const auto* b  = packet.Copy(buffers, count);
		const auto* o  = packet.Copy(offsets, count);
		const auto* sz = packet.Copy(sizes, count);
		const auto* st = packet.Copy(strides, count);
		packet.Commit([=] { g_real.vkCmdBindVertexBuffers2(buffer, first, count, b, o, sz, st); });
		return;
	}
	BeforeDirect();
	g_real.vkCmdBindVertexBuffers2(buffer, first, count, buffers, offsets, sizes, strides);
}

void VKAPI_PTR HookCmdBindIndexBuffer(VkCommandBuffer buffer, VkBuffer index_buffer,
                                      VkDeviceSize offset, VkIndexType type) {
	if (Routed(buffer)) {
		Emit([=] { g_real.vkCmdBindIndexBuffer(buffer, index_buffer, offset, type); });
		return;
	}
	BeforeDirect();
	g_real.vkCmdBindIndexBuffer(buffer, index_buffer, offset, type);
}

void VKAPI_PTR HookCmdBeginRendering(VkCommandBuffer buffer, const VkRenderingInfo* info) {
	if (Routed(buffer)) {
		RequireNoNext(info->pNext, "a rendering info");
		Packet packet(*t_stream, Bytes<VkRenderingInfo>(1) +
		                             Bytes<VkRenderingAttachmentInfo>(info->colorAttachmentCount) +
		                             2 * Bytes<VkRenderingAttachmentInfo>(1));
		auto* copy = packet.Copy(info, 1);
		copy->pColorAttachments = packet.Copy(info->pColorAttachments, info->colorAttachmentCount);
		copy->pDepthAttachment  = packet.Copy(info->pDepthAttachment, 1);
		copy->pStencilAttachment = packet.Copy(info->pStencilAttachment, 1);
		for (uint32_t i = 0; i < info->colorAttachmentCount; i++) {
			RequireNoNext(info->pColorAttachments[i].pNext, "a color attachment");
		}
		packet.Commit([=] { g_real.vkCmdBeginRendering(buffer, copy); });
		return;
	}
	BeforeDirect();
	g_real.vkCmdBeginRendering(buffer, info);
}

void VKAPI_PTR HookCmdEndRendering(VkCommandBuffer buffer) {
	if (Routed(buffer)) {
		Emit([=] { g_real.vkCmdEndRendering(buffer); });
		return;
	}
	BeforeDirect();
	g_real.vkCmdEndRendering(buffer);
}

void VKAPI_PTR HookCmdDraw(VkCommandBuffer buffer, uint32_t vertex_count, uint32_t instance_count,
                           uint32_t first_vertex, uint32_t first_instance) {
	if (Routed(buffer)) {
		Emit([=] {
			g_real.vkCmdDraw(buffer, vertex_count, instance_count, first_vertex, first_instance);
		});
		return;
	}
	BeforeDirect();
	g_real.vkCmdDraw(buffer, vertex_count, instance_count, first_vertex, first_instance);
}

void VKAPI_PTR HookCmdDrawIndexed(VkCommandBuffer buffer, uint32_t index_count,
                                  uint32_t instance_count, uint32_t first_index,
                                  int32_t vertex_offset, uint32_t first_instance) {
	if (Routed(buffer)) {
		Emit([=] {
			g_real.vkCmdDrawIndexed(buffer, index_count, instance_count, first_index, vertex_offset,
			                        first_instance);
		});
		return;
	}
	BeforeDirect();
	g_real.vkCmdDrawIndexed(buffer, index_count, instance_count, first_index, vertex_offset,
	                        first_instance);
}

void VKAPI_PTR HookCmdDrawIndexedIndirect(VkCommandBuffer buffer, VkBuffer args,
                                          VkDeviceSize offset, uint32_t count, uint32_t stride) {
	if (Routed(buffer)) {
		Emit([=] { g_real.vkCmdDrawIndexedIndirect(buffer, args, offset, count, stride); });
		return;
	}
	BeforeDirect();
	g_real.vkCmdDrawIndexedIndirect(buffer, args, offset, count, stride);
}

void VKAPI_PTR HookCmdDrawMeshTasksEXT(VkCommandBuffer buffer, uint32_t x, uint32_t y, uint32_t z) {
	if (Routed(buffer)) {
		Emit([=] { g_real.vkCmdDrawMeshTasksEXT(buffer, x, y, z); });
		return;
	}
	BeforeDirect();
	g_real.vkCmdDrawMeshTasksEXT(buffer, x, y, z);
}

void VKAPI_PTR HookCmdDrawMeshTasksIndirectEXT(VkCommandBuffer buffer, VkBuffer args,
                                               VkDeviceSize offset, uint32_t count,
                                               uint32_t stride) {
	if (Routed(buffer)) {
		Emit([=] { g_real.vkCmdDrawMeshTasksIndirectEXT(buffer, args, offset, count, stride); });
		return;
	}
	BeforeDirect();
	g_real.vkCmdDrawMeshTasksIndirectEXT(buffer, args, offset, count, stride);
}

void VKAPI_PTR HookCmdDispatch(VkCommandBuffer buffer, uint32_t x, uint32_t y, uint32_t z) {
	if (Routed(buffer)) {
		Emit([=] { g_real.vkCmdDispatch(buffer, x, y, z); });
		return;
	}
	BeforeDirect();
	g_real.vkCmdDispatch(buffer, x, y, z);
}

void VKAPI_PTR HookCmdDispatchIndirect(VkCommandBuffer buffer, VkBuffer args, VkDeviceSize offset) {
	if (Routed(buffer)) {
		Emit([=] { g_real.vkCmdDispatchIndirect(buffer, args, offset); });
		return;
	}
	BeforeDirect();
	g_real.vkCmdDispatchIndirect(buffer, args, offset);
}

void VKAPI_PTR HookCmdCopyBuffer(VkCommandBuffer buffer, VkBuffer source, VkBuffer target,
                                 uint32_t count, const VkBufferCopy* regions) {
	if (Routed(buffer)) {
		Packet      packet(*t_stream, Bytes<VkBufferCopy>(count));
		const auto* r = packet.Copy(regions, count);
		packet.Commit([=] { g_real.vkCmdCopyBuffer(buffer, source, target, count, r); });
		return;
	}
	BeforeDirect();
	g_real.vkCmdCopyBuffer(buffer, source, target, count, regions);
}

void VKAPI_PTR HookCmdCopyImage(VkCommandBuffer buffer, VkImage source, VkImageLayout source_layout,
                                VkImage target, VkImageLayout target_layout, uint32_t count,
                                const VkImageCopy* regions) {
	if (Routed(buffer)) {
		Packet      packet(*t_stream, Bytes<VkImageCopy>(count));
		const auto* r = packet.Copy(regions, count);
		packet.Commit([=] {
			g_real.vkCmdCopyImage(buffer, source, source_layout, target, target_layout, count, r);
		});
		return;
	}
	BeforeDirect();
	g_real.vkCmdCopyImage(buffer, source, source_layout, target, target_layout, count, regions);
}

void VKAPI_PTR HookCmdCopyBufferToImage(VkCommandBuffer buffer, VkBuffer source, VkImage target,
                                        VkImageLayout layout, uint32_t count,
                                        const VkBufferImageCopy* regions) {
	if (Routed(buffer)) {
		Packet      packet(*t_stream, Bytes<VkBufferImageCopy>(count));
		const auto* r = packet.Copy(regions, count);
		packet.Commit([=] { g_real.vkCmdCopyBufferToImage(buffer, source, target, layout, count, r); });
		return;
	}
	BeforeDirect();
	g_real.vkCmdCopyBufferToImage(buffer, source, target, layout, count, regions);
}

void VKAPI_PTR HookCmdCopyImageToBuffer(VkCommandBuffer buffer, VkImage source,
                                        VkImageLayout layout, VkBuffer target, uint32_t count,
                                        const VkBufferImageCopy* regions) {
	if (Routed(buffer)) {
		Packet      packet(*t_stream, Bytes<VkBufferImageCopy>(count));
		const auto* r = packet.Copy(regions, count);
		packet.Commit([=] { g_real.vkCmdCopyImageToBuffer(buffer, source, layout, target, count, r); });
		return;
	}
	BeforeDirect();
	g_real.vkCmdCopyImageToBuffer(buffer, source, layout, target, count, regions);
}

void VKAPI_PTR HookCmdResolveImage(VkCommandBuffer buffer, VkImage source,
                                   VkImageLayout source_layout, VkImage target,
                                   VkImageLayout target_layout, uint32_t count,
                                   const VkImageResolve* regions) {
	if (Routed(buffer)) {
		Packet      packet(*t_stream, Bytes<VkImageResolve>(count));
		const auto* r = packet.Copy(regions, count);
		packet.Commit([=] {
			g_real.vkCmdResolveImage(buffer, source, source_layout, target, target_layout, count, r);
		});
		return;
	}
	BeforeDirect();
	g_real.vkCmdResolveImage(buffer, source, source_layout, target, target_layout, count, regions);
}

void VKAPI_PTR HookCmdFillBuffer(VkCommandBuffer buffer, VkBuffer target, VkDeviceSize offset,
                                 VkDeviceSize size, uint32_t data) {
	if (Routed(buffer)) {
		Emit([=] { g_real.vkCmdFillBuffer(buffer, target, offset, size, data); });
		return;
	}
	BeforeDirect();
	g_real.vkCmdFillBuffer(buffer, target, offset, size, data);
}

void VKAPI_PTR HookCmdClearColorImage(VkCommandBuffer buffer, VkImage image, VkImageLayout layout,
                                      const VkClearColorValue* color, uint32_t count,
                                      const VkImageSubresourceRange* ranges) {
	if (Routed(buffer)) {
		Packet      packet(*t_stream,
		                   Bytes<VkClearColorValue>(1) + Bytes<VkImageSubresourceRange>(count));
		const auto* c = packet.Copy(color, 1);
		const auto* r = packet.Copy(ranges, count);
		packet.Commit([=] { g_real.vkCmdClearColorImage(buffer, image, layout, c, count, r); });
		return;
	}
	BeforeDirect();
	g_real.vkCmdClearColorImage(buffer, image, layout, color, count, ranges);
}

void VKAPI_PTR HookCmdClearDepthStencilImage(VkCommandBuffer buffer, VkImage image,
                                             VkImageLayout layout,
                                             const VkClearDepthStencilValue* value, uint32_t count,
                                             const VkImageSubresourceRange* ranges) {
	if (Routed(buffer)) {
		Packet      packet(*t_stream,
		                   Bytes<VkClearDepthStencilValue>(1) + Bytes<VkImageSubresourceRange>(count));
		const auto* v = packet.Copy(value, 1);
		const auto* r = packet.Copy(ranges, count);
		packet.Commit([=] { g_real.vkCmdClearDepthStencilImage(buffer, image, layout, v, count, r); });
		return;
	}
	BeforeDirect();
	g_real.vkCmdClearDepthStencilImage(buffer, image, layout, value, count, ranges);
}

void VKAPI_PTR HookCmdClearAttachments(VkCommandBuffer buffer, uint32_t attachment_count,
                                       const VkClearAttachment* attachments, uint32_t rect_count,
                                       const VkClearRect* rects) {
	if (Routed(buffer)) {
		Packet      packet(*t_stream,
		                   Bytes<VkClearAttachment>(attachment_count) + Bytes<VkClearRect>(rect_count));
		const auto* a = packet.Copy(attachments, attachment_count);
		const auto* r = packet.Copy(rects, rect_count);
		packet.Commit([=] { g_real.vkCmdClearAttachments(buffer, attachment_count, a, rect_count, r); });
		return;
	}
	BeforeDirect();
	g_real.vkCmdClearAttachments(buffer, attachment_count, attachments, rect_count, rects);
}

void VKAPI_PTR HookCmdWriteTimestamp(VkCommandBuffer buffer, VkPipelineStageFlagBits stage,
                                     VkQueryPool pool, uint32_t query) {
	if (Routed(buffer)) {
		Emit([=] { g_real.vkCmdWriteTimestamp(buffer, stage, pool, query); });
		return;
	}
	BeforeDirect();
	g_real.vkCmdWriteTimestamp(buffer, stage, pool, query);
}

void VKAPI_PTR HookCmdResetQueryPool(VkCommandBuffer buffer, VkQueryPool pool, uint32_t first,
                                     uint32_t count) {
	if (Routed(buffer)) {
		Emit([=] { g_real.vkCmdResetQueryPool(buffer, pool, first, count); });
		return;
	}
	BeforeDirect();
	g_real.vkCmdResetQueryPool(buffer, pool, first, count);
}

void VKAPI_PTR HookCmdBeginConditionalRenderingEXT(VkCommandBuffer buffer,
                                                   const VkConditionalRenderingBeginInfoEXT* info) {
	if (Routed(buffer)) {
		RequireNoNext(info->pNext, "a conditional rendering begin");
		Packet      packet(*t_stream, Bytes<VkConditionalRenderingBeginInfoEXT>(1));
		const auto* copy = packet.Copy(info, 1);
		packet.Commit([=] { g_real.vkCmdBeginConditionalRenderingEXT(buffer, copy); });
		return;
	}
	BeforeDirect();
	g_real.vkCmdBeginConditionalRenderingEXT(buffer, info);
}

void VKAPI_PTR HookCmdEndConditionalRenderingEXT(VkCommandBuffer buffer) {
	if (Routed(buffer)) {
		Emit([=] { g_real.vkCmdEndConditionalRenderingEXT(buffer); });
		return;
	}
	BeforeDirect();
	g_real.vkCmdEndConditionalRenderingEXT(buffer);
}

void VKAPI_PTR HookCmdSetViewport(VkCommandBuffer buffer, uint32_t first, uint32_t count,
                                  const VkViewport* viewports) {
	if (Routed(buffer)) {
		Packet      packet(*t_stream, Bytes<VkViewport>(count));
		const auto* v = packet.Copy(viewports, count);
		packet.Commit([=] { g_real.vkCmdSetViewport(buffer, first, count, v); });
		return;
	}
	BeforeDirect();
	g_real.vkCmdSetViewport(buffer, first, count, viewports);
}

void VKAPI_PTR HookCmdSetViewportWithCount(VkCommandBuffer buffer, uint32_t count,
                                           const VkViewport* viewports) {
	if (Routed(buffer)) {
		Packet      packet(*t_stream, Bytes<VkViewport>(count));
		const auto* v = packet.Copy(viewports, count);
		packet.Commit([=] { g_real.vkCmdSetViewportWithCount(buffer, count, v); });
		return;
	}
	BeforeDirect();
	g_real.vkCmdSetViewportWithCount(buffer, count, viewports);
}

void VKAPI_PTR HookCmdSetScissor(VkCommandBuffer buffer, uint32_t first, uint32_t count,
                                 const VkRect2D* scissors) {
	if (Routed(buffer)) {
		Packet      packet(*t_stream, Bytes<VkRect2D>(count));
		const auto* s = packet.Copy(scissors, count);
		packet.Commit([=] { g_real.vkCmdSetScissor(buffer, first, count, s); });
		return;
	}
	BeforeDirect();
	g_real.vkCmdSetScissor(buffer, first, count, scissors);
}

void VKAPI_PTR HookCmdSetScissorWithCount(VkCommandBuffer buffer, uint32_t count,
                                          const VkRect2D* scissors) {
	if (Routed(buffer)) {
		Packet      packet(*t_stream, Bytes<VkRect2D>(count));
		const auto* s = packet.Copy(scissors, count);
		packet.Commit([=] { g_real.vkCmdSetScissorWithCount(buffer, count, s); });
		return;
	}
	BeforeDirect();
	g_real.vkCmdSetScissorWithCount(buffer, count, scissors);
}

void VKAPI_PTR HookCmdSetLineWidth(VkCommandBuffer buffer, float width) {
	if (Routed(buffer)) {
		Emit([=] { g_real.vkCmdSetLineWidth(buffer, width); });
		return;
	}
	BeforeDirect();
	g_real.vkCmdSetLineWidth(buffer, width);
}

void VKAPI_PTR HookCmdSetDepthBias(VkCommandBuffer buffer, float constant, float clamp, float slope) {
	if (Routed(buffer)) {
		Emit([=] { g_real.vkCmdSetDepthBias(buffer, constant, clamp, slope); });
		return;
	}
	BeforeDirect();
	g_real.vkCmdSetDepthBias(buffer, constant, clamp, slope);
}

void VKAPI_PTR HookCmdSetDepthBiasEnable(VkCommandBuffer buffer, VkBool32 enable) {
	if (Routed(buffer)) {
		Emit([=] { g_real.vkCmdSetDepthBiasEnable(buffer, enable); });
		return;
	}
	BeforeDirect();
	g_real.vkCmdSetDepthBiasEnable(buffer, enable);
}

void VKAPI_PTR HookCmdSetBlendConstants(VkCommandBuffer buffer, const float constants[4]) {
	if (Routed(buffer)) {
		const std::array<float, 4> copy {constants[0], constants[1], constants[2], constants[3]};
		Emit([=] { g_real.vkCmdSetBlendConstants(buffer, copy.data()); });
		return;
	}
	BeforeDirect();
	g_real.vkCmdSetBlendConstants(buffer, constants);
}

void VKAPI_PTR HookCmdSetDepthTestEnable(VkCommandBuffer buffer, VkBool32 enable) {
	if (Routed(buffer)) {
		Emit([=] { g_real.vkCmdSetDepthTestEnable(buffer, enable); });
		return;
	}
	BeforeDirect();
	g_real.vkCmdSetDepthTestEnable(buffer, enable);
}

void VKAPI_PTR HookCmdSetDepthWriteEnable(VkCommandBuffer buffer, VkBool32 enable) {
	if (Routed(buffer)) {
		Emit([=] { g_real.vkCmdSetDepthWriteEnable(buffer, enable); });
		return;
	}
	BeforeDirect();
	g_real.vkCmdSetDepthWriteEnable(buffer, enable);
}

void VKAPI_PTR HookCmdSetDepthCompareOp(VkCommandBuffer buffer, VkCompareOp op) {
	if (Routed(buffer)) {
		Emit([=] { g_real.vkCmdSetDepthCompareOp(buffer, op); });
		return;
	}
	BeforeDirect();
	g_real.vkCmdSetDepthCompareOp(buffer, op);
}

void VKAPI_PTR HookCmdSetStencilTestEnable(VkCommandBuffer buffer, VkBool32 enable) {
	if (Routed(buffer)) {
		Emit([=] { g_real.vkCmdSetStencilTestEnable(buffer, enable); });
		return;
	}
	BeforeDirect();
	g_real.vkCmdSetStencilTestEnable(buffer, enable);
}

void VKAPI_PTR HookCmdSetStencilOp(VkCommandBuffer buffer, VkStencilFaceFlags faces,
                                   VkStencilOp fail, VkStencilOp pass, VkStencilOp depth_fail,
                                   VkCompareOp compare) {
	if (Routed(buffer)) {
		Emit([=] { g_real.vkCmdSetStencilOp(buffer, faces, fail, pass, depth_fail, compare); });
		return;
	}
	BeforeDirect();
	g_real.vkCmdSetStencilOp(buffer, faces, fail, pass, depth_fail, compare);
}

void VKAPI_PTR HookCmdSetStencilCompareMask(VkCommandBuffer buffer, VkStencilFaceFlags faces,
                                           uint32_t mask) {
	if (Routed(buffer)) {
		Emit([=] { g_real.vkCmdSetStencilCompareMask(buffer, faces, mask); });
		return;
	}
	BeforeDirect();
	g_real.vkCmdSetStencilCompareMask(buffer, faces, mask);
}

void VKAPI_PTR HookCmdSetStencilWriteMask(VkCommandBuffer buffer, VkStencilFaceFlags faces,
                                         uint32_t mask) {
	if (Routed(buffer)) {
		Emit([=] { g_real.vkCmdSetStencilWriteMask(buffer, faces, mask); });
		return;
	}
	BeforeDirect();
	g_real.vkCmdSetStencilWriteMask(buffer, faces, mask);
}

void VKAPI_PTR HookCmdSetStencilReference(VkCommandBuffer buffer, VkStencilFaceFlags faces,
                                         uint32_t reference) {
	if (Routed(buffer)) {
		Emit([=] { g_real.vkCmdSetStencilReference(buffer, faces, reference); });
		return;
	}
	BeforeDirect();
	g_real.vkCmdSetStencilReference(buffer, faces, reference);
}

void VKAPI_PTR HookCmdSetColorWriteEnableEXT(VkCommandBuffer buffer, uint32_t count,
                                             const VkBool32* enables) {
	if (Routed(buffer)) {
		Packet      packet(*t_stream, Bytes<VkBool32>(count));
		const auto* e = packet.Copy(enables, count);
		packet.Commit([=] { g_real.vkCmdSetColorWriteEnableEXT(buffer, count, e); });
		return;
	}
	BeforeDirect();
	g_real.vkCmdSetColorWriteEnableEXT(buffer, count, enables);
}

void VKAPI_PTR HookCmdSetAttachmentFeedbackLoopEnableEXT(VkCommandBuffer    buffer,
                                                         VkImageAspectFlags aspects) {
	if (Routed(buffer)) {
		Emit([=] { g_real.vkCmdSetAttachmentFeedbackLoopEnableEXT(buffer, aspects); });
		return;
	}
	BeforeDirect();
	g_real.vkCmdSetAttachmentFeedbackLoopEnableEXT(buffer, aspects);
}

// Pool calls run after what every stream holds.
template <typename Tag, typename Fn>
struct DrainingHook;

template <typename Tag, typename R, typename... A>
struct DrainingHook<Tag, R(VKAPI_PTR*)(A...)> {
	static inline R(VKAPI_PTR* real)(A...) = nullptr;

	static R VKAPI_PTR Call(A... args) {
		DrainAll();
		return real(args...);
	}
};

// Other commands: into a routed buffer, after what its stream holds (its consumer then waits
// for more); otherwise like any direct call.
template <typename Tag, typename Fn>
struct FallbackHook;

template <typename Tag, typename R, typename... A>
struct FallbackHook<Tag, R(VKAPI_PTR*)(VkCommandBuffer, A...)> {
	static inline R(VKAPI_PTR* real)(VkCommandBuffer, A...) = nullptr;

	static R VKAPI_PTR Call(VkCommandBuffer buffer, A... args) {
		if (auto* stream = StreamFor(buffer); stream != nullptr) {
			g_fallback_calls.fetch_add(1, std::memory_order_relaxed);
			stream->Drain();
		} else {
			BeforeDirect();
		}
		return real(buffer, args...);
	}
};

// Hook names map to the function names without the "vk" prefix.
#define KYTY_HOOK_FUNCTION_vkBeginCommandBuffer HookBeginCommandBuffer
#define KYTY_HOOK_FUNCTION_vkEndCommandBuffer HookEndCommandBuffer
#define KYTY_HOOK_FUNCTION_vkUpdateDescriptorSets HookUpdateDescriptorSets
#define KYTY_HOOK_FUNCTION_vkQueueSubmit HookQueueSubmit
#define KYTY_HOOK_FUNCTION_vkCmdPipelineBarrier HookCmdPipelineBarrier
#define KYTY_HOOK_FUNCTION_vkCmdPipelineBarrier2 HookCmdPipelineBarrier2
#define KYTY_HOOK_FUNCTION_vkCmdBindPipeline HookCmdBindPipeline
#define KYTY_HOOK_FUNCTION_vkCmdBindDescriptorSets HookCmdBindDescriptorSets
#define KYTY_HOOK_FUNCTION_vkCmdPushDescriptorSetKHR HookCmdPushDescriptorSetKHR
#define KYTY_HOOK_FUNCTION_vkCmdPushConstants HookCmdPushConstants
#define KYTY_HOOK_FUNCTION_vkCmdBindVertexBuffers2 HookCmdBindVertexBuffers2
#define KYTY_HOOK_FUNCTION_vkCmdBindIndexBuffer HookCmdBindIndexBuffer
#define KYTY_HOOK_FUNCTION_vkCmdBeginRendering HookCmdBeginRendering
#define KYTY_HOOK_FUNCTION_vkCmdEndRendering HookCmdEndRendering
#define KYTY_HOOK_FUNCTION_vkCmdDraw HookCmdDraw
#define KYTY_HOOK_FUNCTION_vkCmdDrawIndexed HookCmdDrawIndexed
#define KYTY_HOOK_FUNCTION_vkCmdDrawIndexedIndirect HookCmdDrawIndexedIndirect
#define KYTY_HOOK_FUNCTION_vkCmdDrawMeshTasksEXT HookCmdDrawMeshTasksEXT
#define KYTY_HOOK_FUNCTION_vkCmdDrawMeshTasksIndirectEXT HookCmdDrawMeshTasksIndirectEXT
#define KYTY_HOOK_FUNCTION_vkCmdDispatch HookCmdDispatch
#define KYTY_HOOK_FUNCTION_vkCmdDispatchIndirect HookCmdDispatchIndirect
#define KYTY_HOOK_FUNCTION_vkCmdCopyBuffer HookCmdCopyBuffer
#define KYTY_HOOK_FUNCTION_vkCmdCopyImage HookCmdCopyImage
#define KYTY_HOOK_FUNCTION_vkCmdCopyBufferToImage HookCmdCopyBufferToImage
#define KYTY_HOOK_FUNCTION_vkCmdCopyImageToBuffer HookCmdCopyImageToBuffer
#define KYTY_HOOK_FUNCTION_vkCmdResolveImage HookCmdResolveImage
#define KYTY_HOOK_FUNCTION_vkCmdFillBuffer HookCmdFillBuffer
#define KYTY_HOOK_FUNCTION_vkCmdClearColorImage HookCmdClearColorImage
#define KYTY_HOOK_FUNCTION_vkCmdClearDepthStencilImage HookCmdClearDepthStencilImage
#define KYTY_HOOK_FUNCTION_vkCmdClearAttachments HookCmdClearAttachments
#define KYTY_HOOK_FUNCTION_vkCmdWriteTimestamp HookCmdWriteTimestamp
#define KYTY_HOOK_FUNCTION_vkCmdResetQueryPool HookCmdResetQueryPool
#define KYTY_HOOK_FUNCTION_vkCmdBeginConditionalRenderingEXT HookCmdBeginConditionalRenderingEXT
#define KYTY_HOOK_FUNCTION_vkCmdEndConditionalRenderingEXT HookCmdEndConditionalRenderingEXT
#define KYTY_HOOK_FUNCTION_vkCmdSetViewport HookCmdSetViewport
#define KYTY_HOOK_FUNCTION_vkCmdSetViewportWithCount HookCmdSetViewportWithCount
#define KYTY_HOOK_FUNCTION_vkCmdSetScissor HookCmdSetScissor
#define KYTY_HOOK_FUNCTION_vkCmdSetScissorWithCount HookCmdSetScissorWithCount
#define KYTY_HOOK_FUNCTION_vkCmdSetLineWidth HookCmdSetLineWidth
#define KYTY_HOOK_FUNCTION_vkCmdSetDepthBias HookCmdSetDepthBias
#define KYTY_HOOK_FUNCTION_vkCmdSetDepthBiasEnable HookCmdSetDepthBiasEnable
#define KYTY_HOOK_FUNCTION_vkCmdSetBlendConstants HookCmdSetBlendConstants
#define KYTY_HOOK_FUNCTION_vkCmdSetDepthTestEnable HookCmdSetDepthTestEnable
#define KYTY_HOOK_FUNCTION_vkCmdSetDepthWriteEnable HookCmdSetDepthWriteEnable
#define KYTY_HOOK_FUNCTION_vkCmdSetDepthCompareOp HookCmdSetDepthCompareOp
#define KYTY_HOOK_FUNCTION_vkCmdSetStencilTestEnable HookCmdSetStencilTestEnable
#define KYTY_HOOK_FUNCTION_vkCmdSetStencilOp HookCmdSetStencilOp
#define KYTY_HOOK_FUNCTION_vkCmdSetStencilCompareMask HookCmdSetStencilCompareMask
#define KYTY_HOOK_FUNCTION_vkCmdSetStencilWriteMask HookCmdSetStencilWriteMask
#define KYTY_HOOK_FUNCTION_vkCmdSetStencilReference HookCmdSetStencilReference
#define KYTY_HOOK_FUNCTION_vkCmdSetColorWriteEnableEXT HookCmdSetColorWriteEnableEXT
#define KYTY_HOOK_FUNCTION_vkCmdSetAttachmentFeedbackLoopEnableEXT                              \
	HookCmdSetAttachmentFeedbackLoopEnableEXT

[[nodiscard]] bool Enabled(const char* name) {
	const char* value = std::getenv(name);
	return value != nullptr && std::strcmp(value, "1") == 0;
}

} // namespace

void InstallCommandHooks() {
	static bool installed = false;
	if (installed) {
		return;
	}
	installed        = true;
	auto&       dispatcher = VULKAN_HPP_DEFAULT_DISPATCHER;
	const char* forced     = std::getenv("KYTY_RECORD_THREAD");
	if (forced != nullptr ? std::strcmp(forced, "0") != 0 : Config::RecordThreadEnabled()) {
		g_deferred = true;
#define KYTY_INSTALL_RECORDING_HOOK(name)                                                         \
	g_real.name = dispatcher.name;                                                                \
	if (dispatcher.name != nullptr) {                                                             \
		dispatcher.name = KYTY_HOOK_FUNCTION_##name;                                              \
	}
		KYTY_HOOKED_VK_FUNCTIONS(KYTY_INSTALL_RECORDING_HOOK)
#undef KYTY_INSTALL_RECORDING_HOOK
#define KYTY_INSTALL_DRAINING_HOOK(name)                                                          \
	if (dispatcher.name != nullptr) {                                                             \
		struct Tag_##name {};                                                                     \
		using Hook  = DrainingHook<Tag_##name, PFN_##name>;                                       \
		Hook::real  = dispatcher.name;                                                            \
		g_real.name = dispatcher.name;                                                            \
		dispatcher.name = Hook::Call;                                                             \
	}
		KYTY_DRAINING_VK_FUNCTIONS(KYTY_INSTALL_DRAINING_HOOK)
#undef KYTY_INSTALL_DRAINING_HOOK
#define KYTY_INSTALL_FALLBACK_HOOK(name)                                                          \
	if (dispatcher.name != nullptr) {                                                             \
		struct Tag_##name {};                                                                     \
		using Hook      = FallbackHook<Tag_##name, PFN_##name>;                                   \
		Hook::real      = dispatcher.name;                                                        \
		dispatcher.name = Hook::Call;                                                             \
	}
		KYTY_FALLBACK_VK_FUNCTIONS(KYTY_INSTALL_FALLBACK_HOOK)
#undef KYTY_INSTALL_FALLBACK_HOOK
		return;
	}
	if (Enabled("KYTY_DEBUG_VK_TIME")) {
#define KYTY_INSTALL_TIMED_HOOK(name)                                                             \
	if (dispatcher.name != nullptr) {                                                             \
		struct Tag_##name {};                                                                     \
		using Hook      = TimedHook<Tag_##name, PFN_##name>;                                      \
		Hook::real      = dispatcher.name;                                                        \
		dispatcher.name = Hook::Call;                                                             \
	}
		KYTY_HOOKED_VK_FUNCTIONS(KYTY_INSTALL_TIMED_HOOK)
#undef KYTY_INSTALL_TIMED_HOOK
	}
}

void MarkCommandHookThread() {
	t_timed_thread = true;
}

bool CommandRecordingDeferred() {
	return g_deferred;
}

void DrainGpuThreadCommands() {
	BeforeDirect();
}

CommandStream::CommandStream() {
	for (auto& slot: g_slots) {
		bool expected = false;
		if (slot.in_use.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
			m_slot = &slot;
			break;
		}
	}
	EXIT_IF(m_slot == nullptr);
	m_impl = m_slot->stream.load(std::memory_order_acquire);
	if (m_impl == nullptr) {
		// Kept for the process (see Slot).
		m_impl = new Impl;
		m_slot->stream.store(m_impl, std::memory_order_release);
	}
}

CommandStream::~CommandStream() {
	// The owner stops the consumer after its last submit, so nothing is left.
	EXIT_IF(m_impl->read.load(std::memory_order_acquire) !=
	        m_impl->write.load(std::memory_order_acquire));
	m_slot->buffer.store(VK_NULL_HANDLE, std::memory_order_release);
	m_slot->in_use.store(false, std::memory_order_release);
}

void CommandStream::Route(VkCommandBuffer buffer) {
	// The routing thread keeps its stream between buffers: its direct calls then drain it too.
	t_home = m_slot;
	m_slot->buffer.store(buffer, std::memory_order_release);
}

void CommandStream::Push(Common::UniqueFunction<void>&& call) {
	auto*  heap = new Common::UniqueFunction<void>(std::move(call));
	Packet packet(*m_impl, 0);
	packet.Commit([heap] {
		(*heap)();
		delete heap;
	});
}

void CommandStream::Wake() {
	auto& impl = *m_impl;
	if (impl.sleeping.load(std::memory_order_seq_cst)) {
		impl.Signal();
	}
	static const bool stats = Enabled("KYTY_DEBUG_STREAM_STATS");
	if (stats) {
		using Clock = std::chrono::steady_clock;
		static auto window_start = Clock::now();
		static auto last         = std::array<uint64_t, 6> {};
		const auto  now          = Clock::now();
		if (now - window_start >= std::chrono::seconds(5)) {
			const auto seconds = std::chrono::duration<double>(now - window_start).count();
			const std::array<uint64_t, 6> current {impl.packets.load(), impl.write.load(),
			                                       impl.wakes.load(),   impl.drains.load(),
			                                       impl.sleeps.load(),  g_fallback_calls.load()};
			std::printf("command-stream: %.1fs packets/s=%.0f MB/s=%.1f wakes/s=%.0f drains/s=%.0f "
			            "sleeps/s=%.0f direct/s=%.0f\n",
			            seconds, static_cast<double>(current[0] - last[0]) / seconds,
			            static_cast<double>(current[1] - last[1]) / seconds / 1e6,
			            static_cast<double>(current[2] - last[2]) / seconds,
			            static_cast<double>(current[3] - last[3]) / seconds,
			            static_cast<double>(current[4] - last[4]) / seconds,
			            static_cast<double>(current[5] - last[5]) / seconds);
			last         = current;
			window_start = now;
		}
	}
}

void CommandStream::Drain() {
	m_impl->Drain();
}

void CommandStream::Consume(std::stop_token stop) {
	auto&    impl     = *m_impl;
	uint64_t position = impl.read.load(std::memory_order_relaxed);
	t_consuming       = &impl;
	for (;;) {
		const auto available = impl.write.load(std::memory_order_acquire);
		if (position == available) {
			if (stop.stop_requested()) {
				return;
			}
			// Spin a little: the GPU thread appends a few calls per draw.
			bool more = false;
			for (int i = 0; i < 2000 && !more; i++) {
				_mm_pause();
				more = impl.write.load(std::memory_order_acquire) != position;
			}
			if (more) {
				continue;
			}
			const auto signal = impl.signal.load(std::memory_order_acquire);
			impl.sleeping.store(true, std::memory_order_seq_cst);
			if (impl.write.load(std::memory_order_seq_cst) == position && !stop.stop_requested()) {
				impl.sleeps.fetch_add(1, std::memory_order_relaxed);
				// A stop request only wakes a waiter that re-checks: poll it with a timeout.
				std::stop_callback wake(stop, [&impl] {
					impl.signal.fetch_add(1, std::memory_order_release);
					impl.signal.notify_one();
				});
				impl.signal.wait(signal, std::memory_order_acquire);
			}
			impl.sleeping.store(false, std::memory_order_relaxed);
			continue;
		}
		auto* start  = impl.ring.data() + position % Impl::RingBytes;
		auto* header = reinterpret_cast<Impl::Header*>(start);
		if (header->run == nullptr) {
			// Skip to the ring's start.
			position += Impl::RingBytes - position % Impl::RingBytes;
		} else {
			header->run(start + header->closure_offset);
			position += header->size;
		}
		impl.read.store(position, std::memory_order_release);
	}
}

} // namespace Libs::Graphics
