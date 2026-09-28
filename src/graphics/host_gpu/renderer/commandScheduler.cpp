#include "graphics/host_gpu/renderer/commandScheduler.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandHooks.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/drainStats.h"
#include "graphics/host_gpu/renderer/gpuZones.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <optional>

namespace Libs::Graphics {

static thread_local CommandScheduler* g_deferred_callback_scheduler = nullptr;

namespace {

void ReportVulkanFatal(const char* what, vk::Result result, uint64_t tick, uint32_t debug_op,
                       uint64_t debug_submit, uint32_t arg0, uint32_t arg1, uint32_t arg2,
                       uint32_t arg3, uint64_t arg4) {
	LOGF("%s failed: %s (%d), tick=%" PRIu64 " debug_op=%u debug_submit=%" PRIu64
	     " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
	     what, vk::to_string(result).c_str(), static_cast<int>(result), tick, debug_op,
	     debug_submit, arg0, arg1, arg2, arg3, arg4);
	std::printf("%s failed: %s (%d), tick=%" PRIu64 " debug_op=%u debug_submit=%" PRIu64
	            " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
	            what, vk::to_string(result).c_str(), static_cast<int>(result), tick, debug_op,
	            debug_submit, arg0, arg1, arg2, arg3, arg4);
	std::fflush(stdout);
}

} // namespace

CommandScheduler::CommandPool::CommandPool(GraphicContext& graphics, MasterSemaphore& master)
    : m_graphics(graphics), m_master(master) {
	EXIT_IF(graphics.queue_family == static_cast<uint32_t>(-1));
	vk::CommandPoolCreateInfo create {};
	create.queueFamilyIndex = graphics.queue_family;
	create.flags            = vk::CommandPoolCreateFlagBits::eTransient |
	                          vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
	const auto result       = graphics.device.createCommandPool(&create, nullptr, &m_pool);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_pool == nullptr);
}

CommandScheduler::CommandPool::~CommandPool() {
	m_graphics.device.destroyCommandPool(m_pool, nullptr);
}

size_t CommandScheduler::CommandPool::Grow() {
	const auto first = m_ticks.size();
	m_ticks.resize(first + GrowStep);
	m_buffers.resize(first + GrowStep);

	vk::CommandBufferAllocateInfo allocate {};
	allocate.commandPool        = m_pool;
	allocate.level              = vk::CommandBufferLevel::ePrimary;
	allocate.commandBufferCount = static_cast<uint32_t>(GrowStep);
	EXIT_IF(m_graphics.device.allocateCommandBuffers(&allocate, m_buffers.data() + first) !=
	        vk::Result::eSuccess);
	return first;
}

vk::CommandBuffer CommandScheduler::CommandPool::Commit() {
	auto       gpu_tick = m_master.KnownGpuTick();
	const auto search   = [this, &gpu_tick](size_t begin, size_t end) -> std::optional<size_t> {
		for (size_t index = begin; index < end; ++index) {
			if (gpu_tick >= m_ticks[index]) {
				m_ticks[index] = m_master.CurrentTick();
				return index;
			}
		}
		return std::nullopt;
	};

	auto found = search(m_hint, m_ticks.size());
	if (!found) {
		m_master.Refresh();
		gpu_tick = m_master.KnownGpuTick();
		found    = search(m_hint, m_ticks.size());
	}
	if (!found) {
		found = search(0, m_hint);
	}
	if (!found) {
		found           = Grow();
		m_ticks[*found] = m_master.CurrentTick();
	}

	m_hint = (*found + 1) % m_ticks.size();
	return m_buffers[*found];
}

bool CommandScheduler::InDeferredOperation() noexcept {
	return g_deferred_callback_scheduler != nullptr;
}

CommandScheduler::CommandScheduler(RenderContext& context, GraphicContext& graphics)
    : m_master(graphics), m_context(context), m_graphics(graphics),
      m_command_pool(graphics, m_master), m_command(*this),
      m_priority_thread([this](std::stop_token stop) { PriorityOperationsThread(stop); }) {}

CommandScheduler::~CommandScheduler() {
	Shutdown();
	if (GpuZones::g_context == this) {
		GpuZones::g_marker  = nullptr;
		GpuZones::g_context = nullptr;
	}
	if (m_timestamp_pool != nullptr) {
		m_graphics.device.destroyQueryPool(m_timestamp_pool, nullptr);
	}
	if (m_zone_pool != nullptr) {
		m_graphics.device.destroyQueryPool(m_zone_pool, nullptr);
	}
}

void CommandScheduler::Shutdown() {
	{
		std::unique_lock lock(m_operation_mutex);
		if (m_operation_state == OperationState::Closed) {
			return;
		}
		if (g_deferred_callback_scheduler == this) {
			EXIT_IF(m_operation_state == OperationState::Open);
			// A priority callback cannot join its own runner, while a normal callback can be
			// executing inside the shutdown owner's final PopPendingOperations. The owning
			// thread will finish shutdown after this callback returns.
			return;
		}
		if (m_operation_state == OperationState::Draining) {
			m_operation_available.wait(
			    lock, [this] { return m_operation_state == OperationState::Closed; });
			return;
		}
		m_operation_state = OperationState::Draining;
	}
	if (!m_command.IsInvalid()) {
		Submit();
	}
	m_master.Wait(CurrentTick() - 1);
	PopPendingOperations();
	DrainPriorityOperations();
	m_priority_thread.request_stop();
	m_operation_available.notify_all();
	if (m_priority_thread.joinable()) {
		m_priority_thread.join();
	}
	// Every tick was waited on above, so the submit thread has nothing left to submit.
	StopSubmitThread();
	{
		std::lock_guard lock(m_operation_mutex);
		EXIT_IF(!m_pending_operations.empty() || !m_priority_operations.empty() ||
		        m_priority_active);
		m_operation_state = OperationState::Closed;
	}
	m_operation_available.notify_all();
}

void CommandScheduler::Begin(HW::Context& registers, HW::UserConfig& user_config,
                             HW::Shader& shaders) {
	{
		std::lock_guard lock(m_operation_mutex);
		EXIT_IF(m_operation_state != OperationState::Open);
	}
	m_command.Bind(registers, user_config, shaders);

	if (m_command.IsInvalid()) {
		BeginNext();
	}
}

void CommandScheduler::BeginRendering(const RenderState& state) {
	Current().BeginRendering(state);
}

void CommandScheduler::EndRendering() {
	if (Active() && !m_command.IsInvalid()) {
		Current().EndRendering();
	}
}

void CommandScheduler::Flush() {
	SubmitInfo submit;
	Flush(submit);
}

void CommandScheduler::Flush(SubmitInfo& submit) {
	Submit(submit);
	BeginNext();
}

void CommandScheduler::FlushAndWait() {
	DrainStats::WaitTimer timer(DrainStats::Kind::FullDrain);
	const auto            tick = Submit();
	m_master.Wait(tick);
	BeginNext();
}

void CommandScheduler::Finish() {
	CheckActive();
	{
		DrainStats::WaitTimer timer(DrainStats::Kind::FullDrain);
		if (!m_command.IsInvalid()) {
			Submit();
		}
		m_master.Wait(CurrentTick() - 1);
		BeginNext();
	}
	PopPendingOperations();
}

void CommandScheduler::Wait(uint64_t tick) {
	EXIT_IF(tick > CurrentTick());
	if (tick == CurrentTick()) {
		CheckActive();
		DrainStats::WaitTimer timer(DrainStats::Kind::FullDrain);
		// A stream-buffer wrap can wait while a draw is being prepared through a reference to
		// Current(). The wrapper stays stable while its pooled Vulkan buffer is retired. Deferred
		// resources are released only at the next GPU operation boundary.
		const auto submitted_tick = Submit();
		EXIT_IF(submitted_tick != tick);
		m_master.Wait(tick);
		BeginNext();
	} else if (DrainStats::Enabled() && !IsFree(tick)) {
		DrainStats::WaitTimer timer(DrainStats::Kind::TickWait);
		m_master.Wait(tick);
	} else {
		m_master.Wait(tick);
	}
}

void CommandScheduler::PopPendingOperations() {
	// Draws arrive every few microseconds, the GPU completes submissions every few milliseconds,
	// and no one waits on these operations (deletions, recycling, fault processing): query the
	// GPU's progress at most once per interval. IsFree and the command pool query on demand.
	static constexpr auto RefreshInterval = std::chrono::microseconds(200);
	static const bool     ab              = AbSelected("pending");
	const auto            now             = std::chrono::steady_clock::now();
	if ((ab && AbFeatureOff()) || now - m_last_pending_refresh >= RefreshInterval) {
		m_master.Refresh();
		m_last_pending_refresh = now;
	}
	for (;;) {
		PendingOperation operation;
		{
			std::lock_guard lock(m_operation_mutex);
			if (m_pending_operations.empty() ||
			    !m_master.IsFree(m_pending_operations.front().tick)) {
				return;
			}
			operation = std::move(m_pending_operations.front());
			m_pending_operations.pop();
		}
		WaitPriorityOperations(operation.tick);
		RunOperation(std::move(operation.callback));
	}
}

void CommandScheduler::DeferOperation(Common::UniqueFunction<void>&& operation) {
	CheckActive();
	EXIT_IF(!operation);
	std::unique_lock lock(m_operation_mutex);
	if (m_operation_state == OperationState::Open) {
		m_pending_operations.push({std::move(operation), CurrentTick()});
		return;
	}
	if (g_deferred_callback_scheduler == this) {
		lock.unlock();
		operation();
		return;
	}
	m_operation_available.wait(lock,
	                           [this] { return m_operation_state == OperationState::Closed; });
	lock.unlock();
	operation();
}

void CommandScheduler::DeferPriorityOperation(Common::UniqueFunction<void>&& operation) {
	CheckActive();
	EXIT_IF(!operation);
	std::unique_lock lock(m_operation_mutex);
	if (m_operation_state == OperationState::Open) {
		m_priority_operations.push({std::move(operation), CurrentTick()});
		lock.unlock();
		m_operation_available.notify_one();
		return;
	}
	if (g_deferred_callback_scheduler == this) {
		lock.unlock();
		operation();
		return;
	}
	m_operation_available.wait(lock,
	                           [this] { return m_operation_state == OperationState::Closed; });
	lock.unlock();
	operation();
}

void CommandScheduler::PriorityOperationsThread(std::stop_token stop) {
	while (!stop.stop_requested()) {
		PendingOperation operation;
		{
			std::unique_lock lock(m_operation_mutex);
			m_operation_available.wait(lock, [this, &stop] {
				return stop.stop_requested() || !m_priority_operations.empty();
			});
			if (stop.stop_requested()) {
				return;
			}
			operation = std::move(m_priority_operations.front());
			m_priority_operations.pop();
			m_priority_active      = true;
			m_priority_active_tick = operation.tick;
		}
		m_master.Wait(operation.tick);
		if (!stop.stop_requested()) {
			RunOperation(std::move(operation.callback));
		}
		{
			std::lock_guard lock(m_operation_mutex);
			m_priority_active      = false;
			m_priority_active_tick = 0;
		}
		m_operation_available.notify_all();
	}
}

void CommandScheduler::DrainPriorityOperations() {
	EXIT_IF(g_deferred_callback_scheduler == this);
	std::unique_lock lock(m_operation_mutex);
	m_operation_available.wait(
	    lock, [this] { return m_priority_operations.empty() && !m_priority_active; });
}

void CommandScheduler::WaitPriorityOperations(uint64_t tick) {
	EXIT_IF(g_deferred_callback_scheduler == this);
	std::unique_lock lock(m_operation_mutex);
	const auto       done = [this, tick] {
		const bool active_before_or_at = m_priority_active && m_priority_active_tick <= tick;
		const bool queued_before_or_at =
		    !m_priority_operations.empty() && m_priority_operations.front().tick <= tick;
		return !active_before_or_at && !queued_before_or_at;
	};
	if (done()) {
		return;
	}
	DrainStats::WaitTimer timer(DrainStats::Kind::PriorityWait);
	m_operation_available.wait(lock, done);
}

void CommandScheduler::RunOperation(Common::UniqueFunction<void>&& operation) {
	auto* previous                = g_deferred_callback_scheduler;
	g_deferred_callback_scheduler = this;
	operation();
	g_deferred_callback_scheduler = previous;
}

bool CommandScheduler::IsPublished(uint64_t tick) {
	if (!IsFree(tick)) {
		return false;
	}
	std::lock_guard lock(m_operation_mutex);
	const bool queued = !m_priority_operations.empty() && m_priority_operations.front().tick <= tick;
	const bool active = m_priority_active && m_priority_active_tick <= tick;
	return !queued && !active;
}

uint64_t CommandScheduler::MicrosSinceSubmit() const noexcept {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
	                                 std::chrono::steady_clock::now() - m_last_submit)
	                                 .count());
}

bool CommandScheduler::IsFree(uint64_t tick) {
	if (m_master.IsFree(tick)) {
		return true;
	}
	m_master.Refresh();
	return m_master.IsFree(tick);
}

void CommandScheduler::CheckActive() const {
	EXIT_IF(!Active());
}

CommandBuffer& CommandScheduler::Current() {
	CheckActive();
	return m_command;
}

CommandBuffer& CommandScheduler::BeginCommand() {
	EXIT_IF(!m_command.IsInvalid());
	m_command.m_buffer = m_command_pool.Commit();
	if (m_async_submit) {
		MarkCommandHookThread();
	}
	if (m_stream != nullptr) {
		// Its Begin and every command the GPU thread records into it go through the stream.
		// KYTY_DEBUG_AB=recorder records directly in every other window (after draining).
		static const bool ab    = AbSelected("recorder");
		const bool        route = !(ab && AbFeatureOff());
		m_stream->Route(route ? m_command.m_buffer : VK_NULL_HANDLE);
	}
	m_command.Begin();
	if (m_async_submit && DrainStats::Enabled()) {
		WriteStartTimestamp();
		if (m_zones) {
			BeginZones();
		}
	}
	return m_command;
}

void CommandScheduler::WriteStartTimestamp() {
	if (m_timestamp_pool == nullptr) {
		vk::QueryPoolCreateInfo info {};
		info.queryType  = vk::QueryType::eTimestamp;
		info.queryCount = TimestampSlots * 2;
		RequireVulkanSuccess(m_graphics.device.createQueryPool(&info, nullptr, &m_timestamp_pool),
		                     "create GPU timestamp pool");
	}
	m_timestamp_slot = m_timestamp_next++ % TimestampSlots;
	m_command.m_buffer.resetQueryPool(m_timestamp_pool, m_timestamp_slot * 2, 2);
	m_command.m_buffer.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, m_timestamp_pool,
	                                  m_timestamp_slot * 2);
}

void CommandScheduler::WriteEndTimestamp() {
	const auto slot  = m_timestamp_slot;
	m_timestamp_slot = UINT32_MAX;
	m_command.m_buffer.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, m_timestamp_pool,
	                                  slot * 2 + 1);
	// Runs on this thread once the buffer's tick completes, in tick order.
	std::lock_guard lock(m_operation_mutex);
	m_pending_operations.push({[this, slot] { ReadTimestamps(slot); }, CurrentTick()});
}

void CommandScheduler::MarkZoneThunk(void* context, vk::CommandBuffer buffer,
                                     DrainStats::Zone zone, uint64_t key, uint64_t pixels) {
	auto* self = static_cast<CommandScheduler*>(context);
	if (self->m_zone_chunk != UINT32_MAX && buffer == self->m_command.m_buffer) {
		self->MarkZone(zone, key, pixels);
	}
}

void CommandScheduler::BeginZones() {
	if (m_zone_pool == nullptr) {
		vk::QueryPoolCreateInfo info {};
		info.queryType  = vk::QueryType::eTimestamp;
		info.queryCount = ZoneChunkQueries * ZoneChunks;
		RequireVulkanSuccess(m_graphics.device.createQueryPool(&info, nullptr, &m_zone_pool),
		                     "create GPU zone timestamp pool");
	}
	m_zone_chunk = m_zone_next_chunk++ % ZoneChunks;
	m_command.m_buffer.resetQueryPool(m_zone_pool, m_zone_chunk * ZoneChunkQueries,
	                                  ZoneChunkQueries);
	m_zone_marks.clear();
	MarkZone(DrainStats::Zone::Unmarked, 0);
}

void CommandScheduler::MarkZone(DrainStats::Zone zone, uint64_t key, uint64_t pixels) {
	if (!m_zone_marks.empty() && m_zone_marks.back().zone == zone &&
	    m_zone_marks.back().key == key) {
		return;
	}
	// The chunk's last query closes the buffer; work past a full chunk joins the current zone.
	if (m_zone_marks.size() + 1 >= ZoneChunkQueries) {
		return;
	}
	// Bottom of pipe: the timestamp lands once the work recorded before it has finished.
	m_command.m_buffer.writeTimestamp(
	    vk::PipelineStageFlagBits::eBottomOfPipe, m_zone_pool,
	    m_zone_chunk * ZoneChunkQueries + static_cast<uint32_t>(m_zone_marks.size()));
	m_zone_marks.push_back({zone, key, pixels});
}

void CommandScheduler::EndZones() {
	if (m_zone_chunk == UINT32_MAX) {
		return;
	}
	const auto chunk = m_zone_chunk;
	m_zone_chunk     = UINT32_MAX;
	m_command.m_buffer.writeTimestamp(
	    vk::PipelineStageFlagBits::eBottomOfPipe, m_zone_pool,
	    chunk * ZoneChunkQueries + static_cast<uint32_t>(m_zone_marks.size()));
	std::lock_guard lock(m_operation_mutex);
	m_pending_operations.push(
	    {[this, chunk, marks = std::move(m_zone_marks)] { ReadZones(chunk, marks); },
	     CurrentTick()});
	m_zone_marks = {};
}

void CommandScheduler::ReadZones(uint32_t chunk, const std::vector<ZoneMark>& marks) {
	if (marks.empty()) {
		return;
	}
	std::vector<uint64_t> values(marks.size() + 1);
	const auto            result = m_graphics.device.getQueryPoolResults(
        m_zone_pool, chunk * ZoneChunkQueries, static_cast<uint32_t>(values.size()),
        values.size() * sizeof(uint64_t), values.data(), sizeof(uint64_t),
        vk::QueryResultFlagBits::e64);
	if (result != vk::Result::eSuccess) {
		return;
	}
	const double period = m_graphics.GetPhysicalDeviceProperties().limits.timestampPeriod;
	std::vector<DrainStats::ZoneSample> samples;
	samples.reserve(marks.size());
	for (size_t i = 0; i < marks.size(); i++) {
		// A later buffer that reused this chunk before the read leaves values out of order.
		if (values[i + 1] < values[i]) {
			return;
		}
		samples.push_back(
		    {marks[i].zone, marks[i].key, marks[i].pixels,
		     static_cast<uint64_t>(static_cast<double>(values[i + 1] - values[i]) * period)});
	}
	DrainStats::RecordZones(samples.data(), samples.size());
}

void CommandScheduler::ReadTimestamps(uint32_t slot) {
	uint64_t   values[2] {};
	const auto result = m_graphics.device.getQueryPoolResults(
	    m_timestamp_pool, slot * 2, 2, sizeof(values), values, sizeof(uint64_t),
	    vk::QueryResultFlagBits::e64);
	if (result != vk::Result::eSuccess || values[1] < values[0]) {
		return;
	}
	const double period = m_graphics.GetPhysicalDeviceProperties().limits.timestampPeriod;
	const auto   to_ns  = [period](uint64_t ticks) {
        return static_cast<uint64_t>(static_cast<double>(ticks) * period);
	};
	// Busy time is the union of command-buffer intervals; a gap is GPU time with no buffer.
	const auto [start, end] = std::pair {values[0], values[1]};
	if (m_gpu_last_end != 0 && start > m_gpu_last_end) {
		DrainStats::Record(DrainStats::Kind::GpuGap, to_ns(start - m_gpu_last_end));
	}
	const auto from = std::max(start, m_gpu_last_end);
	if (end > from) {
		DrainStats::Record(DrainStats::Kind::GpuBusy, to_ns(end - from));
		g_gpu_busy_ns.fetch_add(to_ns(end - from), std::memory_order_relaxed);
	}
	m_gpu_last_end = std::max(m_gpu_last_end, end);
}

uint64_t CommandScheduler::Submit(SubmitInfo submit) {
	EXIT_IF(m_command.IsInvalid());
	EXIT_IF(submit.num_wait_semaphores > SubmitInfo::MaxSemaphores ||
	        submit.num_signal_semaphores >= SubmitInfo::MaxSemaphores);

	if (m_timestamp_slot != UINT32_MAX) {
		WriteEndTimestamp();
	}
	EndZones();
	m_last_submit = std::chrono::steady_clock::now();
	m_command.End();
	EXIT_IF(m_graphics.queue == nullptr);

	SubmitJob job {
	    .buffer       = m_command.m_buffer,
	    .submit       = submit,
	    .debug_op     = m_command.m_debug_op,
	    .debug_submit = m_command.m_debug_submit_id,
	    .debug_arg0   = m_command.m_debug_arg0,
	    .debug_arg1   = m_command.m_debug_arg1,
	    .debug_arg2   = m_command.m_debug_arg2,
	    .debug_arg3   = m_command.m_debug_arg3,
	    .debug_arg4   = m_command.m_debug_arg4,
	    .reason       = static_cast<uint8_t>(DrainStats::CurrentReason()),
	    .pm4_op       = DrainStats::t_pm4_op,
	};
	m_command.m_buffer = nullptr;
	if (!m_async_submit) {
		QueueSubmit(job);
		return job.tick;
	}
	if (m_stream != nullptr) {
		// The submit thread records the stream's commands and submits in order.
		job.tick = m_master.NextTick();
		job.submit.AddSignal(m_master.Handle(), job.tick);
		m_stream->Route(VK_NULL_HANDLE);
		m_stream->Push([this, job]() mutable { QueueSubmit(job); });
		m_stream->Wake();
		return job.tick;
	}
	{
		// Ticks are allocated in queue order, so the timeline is signaled in order.
		std::lock_guard lock(m_submit_mutex);
		job.tick = m_master.NextTick();
		job.submit.AddSignal(m_master.Handle(), job.tick);
		m_submit_jobs.push_back(job);
	}
	m_submit_available.notify_one();
	return job.tick;
}

void CommandScheduler::QueueSubmit(SubmitJob& job) {
	auto&      graphics     = m_graphics;
	const auto reason       = static_cast<DrainStats::Reason>(job.reason);
	const bool stats        = DrainStats::Enabled();
	const auto submit_start = stats ? std::chrono::steady_clock::now()
	                                : std::chrono::steady_clock::time_point {};
	const auto elapsed_ns   = [&submit_start] {
		return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                                 std::chrono::steady_clock::now() - submit_start)
		                                 .count());
	};
	vk::Result result;
	if (!m_async_submit) {
		// After what the GPU thread's command stream still holds, and outside the queue lock.
		DrainGpuThreadCommands();
	}
	{
		Common::LockGuard lock(graphics.queue_mutex);
		if (stats) {
			DrainStats::Record(DrainStats::Kind::QueueLockWait, reason, job.pm4_op, elapsed_ns());
		}
		if (!m_async_submit) {
			job.tick = m_master.NextTick();
			job.submit.AddSignal(m_master.Handle(), job.tick);
		}
		auto& submit = job.submit;

		vk::TimelineSemaphoreSubmitInfo timeline_info {};
		timeline_info.waitSemaphoreValueCount   = submit.num_wait_semaphores;
		timeline_info.pWaitSemaphoreValues      = submit.wait_ticks.data();
		timeline_info.signalSemaphoreValueCount = submit.num_signal_semaphores;
		timeline_info.pSignalSemaphoreValues    = submit.signal_ticks.data();

		vk::SubmitInfo submit_info {};
		submit_info.pNext                = &timeline_info;
		submit_info.waitSemaphoreCount   = submit.num_wait_semaphores;
		submit_info.pWaitSemaphores      = submit.wait_semaphores.data();
		submit_info.pWaitDstStageMask    = submit.wait_stages.data();
		submit_info.commandBufferCount   = 1;
		submit_info.pCommandBuffers      = &job.buffer;
		submit_info.signalSemaphoreCount = submit.num_signal_semaphores;
		submit_info.pSignalSemaphores    = submit.signal_semaphores.data();

		result = graphics.queue.submit(1, &submit_info, nullptr);
	}
	if (stats) {
		DrainStats::Record(DrainStats::Kind::Submit, reason, job.pm4_op, elapsed_ns());
	}

	if (result != vk::Result::eSuccess) {
		ReportVulkanFatal("vkQueueSubmit", result, job.tick, job.debug_op, job.debug_submit,
		                  job.debug_arg0, job.debug_arg1, job.debug_arg2, job.debug_arg3,
		                  job.debug_arg4);
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

void CommandScheduler::EnableAsyncSubmit() {
	// Switching needs no other submitter running; the owner enables it during construction.
	EXIT_IF(m_async_submit);
	m_async_submit  = true;
	InstallCommandHooks();
	if (CommandRecordingDeferred()) {
		m_stream = std::make_unique<CommandStream>();
	}
	// Only the render scheduler submits asynchronously, so it alone owns the zone marker.
	if (const char* zones = std::getenv("KYTY_GPU_ZONES");
	    zones != nullptr && zones[0] == '1' && GpuZones::g_marker == nullptr) {
		m_zones             = true;
		GpuZones::g_context = this;
		GpuZones::g_marker  = &MarkZoneThunk;
	}
	m_submit_thread = std::jthread([this](std::stop_token stop) { SubmitThread(stop); });
}

void CommandScheduler::SubmitThread(std::stop_token stop) {
	KYTY_PROFILER_THREAD("GpuQueueSubmit");
	if (m_stream != nullptr) {
		m_stream->Consume(stop);
		return;
	}
	for (;;) {
		SubmitJob job;
		{
			std::unique_lock lock(m_submit_mutex);
			// A stop request still drains the queued jobs: their ticks may already be waited on.
			if (!m_submit_available.wait(lock, stop, [this] { return !m_submit_jobs.empty(); })) {
				return;
			}
			job = m_submit_jobs.front();
			m_submit_jobs.pop_front();
		}
		QueueSubmit(job);
	}
}

void CommandScheduler::StopSubmitThread() {
	if (m_submit_thread.joinable()) {
		m_submit_thread.request_stop();
		m_submit_thread.join();
	}
	std::lock_guard lock(m_submit_mutex);
	EXIT_IF(!m_submit_jobs.empty());
}

void CommandScheduler::BeginNext() {
	CheckActive();
	BeginCommand();
}

} // namespace Libs::Graphics
