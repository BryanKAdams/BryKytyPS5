#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_

#include "common/common.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/renderer/drainStats.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/renderer/render.h"

#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>

#include <queue>

#include <thread>
#include <vector>

namespace Libs::Graphics {

class CommandScheduler {
public:
	CommandScheduler(RenderContext& context, GraphicContext& graphics);
	~CommandScheduler();
	KYTY_CLASS_NO_COPY(CommandScheduler);

	void           Begin(HW::Context& registers, HW::UserConfig& user_config, HW::Shader& shaders);
	void           BeginRendering(const RenderState& state);
	void           EndRendering();
	void           Flush();
	void           Flush(SubmitInfo& submit);
	void           FlushAndWait();
	void           Finish();
	CommandBuffer& BeginCommand();
	uint64_t       Submit(SubmitInfo submit = {});
	// Deferred callbacks can observe an externally owned drain, but cannot initiate shutdown:
	// the priority runner cannot join itself.
	void                      Shutdown();
	void                      Wait(uint64_t tick);
	void                      PopPendingOperations();
	void                      DrainPriorityOperations();
	void                      WaitPriorityOperations(uint64_t tick);
	// Whether `tick` completed and every priority operation deferred up to it has run.
	[[nodiscard]] bool        IsPublished(uint64_t tick);
	void                      DeferOperation(Common::UniqueFunction<void>&& operation);
	void                      DeferPriorityOperation(Common::UniqueFunction<void>&& operation);
	[[nodiscard]] static bool InDeferredOperation() noexcept;
	// Hands later submissions to a dedicated queue thread, which submits them in tick order.
	// Submit() then returns once the tick is allocated. Host waits on the timeline stay valid
	// before the submission happens. Enable before the first Submit().
	void EnableAsyncSubmit();

	[[nodiscard]] bool Active() const noexcept { return m_command.m_registers != nullptr; }
	void                           CheckActive() const;
	CommandBuffer&                 Current();
	[[nodiscard]] uint64_t         CurrentTick() const noexcept { return m_master.CurrentTick(); }
	[[nodiscard]] bool             IsFree(uint64_t tick);
	// Microseconds since this scheduler last submitted (recording thread only).
	[[nodiscard]] uint64_t         MicrosSinceSubmit() const noexcept;
	[[nodiscard]] MasterSemaphore& GetMasterSemaphore() noexcept { return m_master; }
	[[nodiscard]] RenderContext&   Context() const noexcept { return m_context; }
	[[nodiscard]] GraphicContext&  Graphics() const noexcept { return m_graphics; }

private:
	class CommandPool {
	public:
		CommandPool(GraphicContext& graphics, MasterSemaphore& master);
		~CommandPool();
		KYTY_CLASS_NO_COPY(CommandPool);

		vk::CommandBuffer Commit();

	private:
		static constexpr size_t GrowStep = 4;

		size_t Grow();

		GraphicContext&                m_graphics;
		MasterSemaphore&               m_master;
		vk::CommandPool                m_pool = nullptr;
		std::vector<vk::CommandBuffer> m_buffers;
		std::vector<uint64_t>          m_ticks;
		size_t                         m_hint = 0;
	};

	enum class OperationState { Open, Draining, Closed };

	struct PendingOperation {
		Common::UniqueFunction<void> callback;
		uint64_t                     tick = 0;
	};

	// A finished command buffer and everything its vkQueueSubmit needs, including the debug
	// state for fatal reports and the drain-stats attribution of the submitting thread.
	struct SubmitJob {
		vk::CommandBuffer buffer = nullptr;
		SubmitInfo        submit;
		uint64_t          tick         = 0;
		uint32_t          debug_op     = 0;
		uint64_t          debug_submit = 0;
		uint32_t          debug_arg0   = 0;
		uint32_t          debug_arg1   = 0;
		uint32_t          debug_arg2   = 0;
		uint32_t          debug_arg3   = 0;
		uint64_t          debug_arg4   = 0;
		uint8_t           reason       = 0;
		uint32_t          pm4_op       = 0;
	};

	void BeginNext();
	void PriorityOperationsThread(std::stop_token stop);
	void SubmitThread(std::stop_token stop);
	void QueueSubmit(SubmitJob& job);
	void StopSubmitThread();
	void RunOperation(Common::UniqueFunction<void>&& operation);

	MasterSemaphore              m_master;
	RenderContext&               m_context;
	GraphicContext&              m_graphics;
	CommandPool                  m_command_pool;
	CommandBuffer                m_command;
	std::queue<PendingOperation> m_pending_operations;
	std::queue<PendingOperation> m_priority_operations;
	std::mutex                   m_operation_mutex;
	std::condition_variable      m_operation_available;
	std::jthread                 m_priority_thread;
	bool                         m_priority_active      = false;
	uint64_t                     m_priority_active_tick = 0;
	OperationState               m_operation_state      = OperationState::Open;
	// Asynchronous submission; jobs are queued in tick order by the recording thread.
	std::mutex                   m_submit_mutex;
	std::condition_variable_any  m_submit_available;
	std::deque<SubmitJob>        m_submit_jobs;
	std::jthread                 m_submit_thread;
	bool                         m_async_submit = false;
	// --drain-stats on the render scheduler: GPU execution time from timestamps written at the
	// start and end of each command buffer, read back once its tick completes.
	static constexpr uint32_t TimestampSlots = 4096;
	void                      WriteStartTimestamp();
	void                      WriteEndTimestamp();
	void                      ReadTimestamps(uint32_t slot);
	vk::QueryPool             m_timestamp_pool = nullptr;
	uint32_t                  m_timestamp_next = 0;
	uint32_t                  m_timestamp_slot = UINT32_MAX; // Slot of the recording buffer.
	uint64_t                  m_gpu_last_end   = 0;          // Latest end seen, in ticks.
	// KYTY_GPU_ZONES=1 with --drain-stats: a timestamp at each change of zone (gpuZones.h) in
	// the recording buffer. Each buffer takes one chunk of the pool, reset when it begins, and
	// its intervals are read once its tick completes.
	static constexpr uint32_t ZoneChunkQueries = 4096;
	static constexpr uint32_t ZoneChunks       = 64;
	struct ZoneMark {
		DrainStats::Zone zone;
		uint64_t         key;
		uint64_t         pixels; // Render area of the interval's first draw.
	};
	static void MarkZoneThunk(void* context, vk::CommandBuffer buffer, DrainStats::Zone zone,
	                          uint64_t key, uint64_t pixels);
	void        BeginZones();
	void        MarkZone(DrainStats::Zone zone, uint64_t key, uint64_t pixels = 0);
	void        EndZones();
	void        ReadZones(uint32_t chunk, const std::vector<ZoneMark>& marks);
	bool                  m_zones           = false;
	vk::QueryPool         m_zone_pool       = nullptr;
	uint32_t              m_zone_next_chunk = 0;
	uint32_t              m_zone_chunk      = UINT32_MAX; // Chunk of the recording buffer.
	std::vector<ZoneMark> m_zone_marks;
	std::chrono::steady_clock::time_point m_last_submit {};
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
