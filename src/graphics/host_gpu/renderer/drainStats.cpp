#include "graphics/host_gpu/renderer/drainStats.h"

#include "common/logging/log.h"
#include "kernel/pthread.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <fmt/format.h>
#include <map>
#include <mutex>
#include <stop_token>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics::DrainStats {

namespace {

constexpr size_t KindCount   = static_cast<size_t>(Kind::Count);
constexpr size_t ReasonCount = static_cast<size_t>(Reason::Count);
constexpr size_t OpCount     = Pm4OpCount + 1;
constexpr size_t CellCount   = KindCount * ReasonCount * OpCount;

struct Cell {
	std::atomic<uint64_t> count {0};
	std::atomic<uint64_t> value {0};
};

constexpr size_t FrameBuckets = 101;

struct Snapshot {
	std::array<uint64_t, FrameBuckets> frame_ms {};
	std::vector<uint64_t> count = std::vector<uint64_t>(CellCount);
	std::vector<uint64_t> value = std::vector<uint64_t>(CellCount);
	uint64_t              frames   = 0;
	uint64_t              presents = 0;
};

Cell                  g_cells[CellCount];
// Everything recorded since the last new game frame, by kind, for the hitch report.
Cell                  g_frame_cells[KindCount];
std::atomic<uint64_t> g_frames {0};
std::atomic<uint64_t> g_presents {0};
// Time between new game frames: one bucket per millisecond, the last collects the rest.
std::array<std::atomic<uint64_t>, FrameBuckets> g_frame_ms {};

struct FaultSite {
	std::string thread;
	uint64_t    count        = 0;
	uint64_t    ns           = 0;
	uint64_t    last_address = 0;
	bool        write        = false;
	// Read faults: the newest recorded GPU writer of the faulting address.
	uint64_t    written      = 0; // Faults whose writer is still in the history.
	uint64_t    frame_age    = 0; // Sum of frames between the write and the fault.
	uint32_t    writer_op    = NoPm4Op;
	uint64_t    writer_begin = 0; // Range of the newest writer seen.
	uint64_t    writer_size  = 0;
};

// Per-interval fault sites, keyed by faulting instruction and access. Faults that stall are
// rare (a few per frame), so a mutex is cheap here.
std::mutex                              g_site_mutex;
std::unordered_map<uint64_t, FaultSite> g_sites;

struct GpuWrite {
	uint64_t begin = 0;
	uint64_t end   = 0;
	uint64_t frame = 0;
	uint32_t op    = NoPm4Op;
};

// Recent GPU write ranges, newest last. Only read faults search it.
constexpr size_t      WriteHistory = 16384;
std::mutex            g_write_mutex;
std::vector<GpuWrite> g_writes(WriteHistory);
size_t                g_write_next = 0;

struct ZoneCell {
	uint64_t count  = 0;
	uint64_t ns     = 0;
	uint64_t pixels = 0; // Summed render area of the intervals that report one.
};

// Per-interval zone time, keyed by zone and key. One command buffer's samples arrive together.
std::mutex                                    g_zone_mutex;
std::map<std::pair<Zone, uint64_t>, ZoneCell> g_zones;

std::mutex                  g_reporter_mutex;
std::condition_variable_any g_reporter_wake;
std::jthread                g_reporter;
// Only an explicit Stop() prints the session total; static destruction may follow Log shutdown.
std::atomic_bool            g_final_report {false};

constexpr size_t Index(Kind kind, Reason reason, uint32_t op) {
	return (static_cast<size_t>(kind) * ReasonCount + static_cast<size_t>(reason)) * OpCount +
	       std::min<size_t>(op, NoPm4Op);
}

const char* KindName(Kind kind) {
	switch (kind) {
		case Kind::FullDrain: return "full-drain";
		case Kind::TickWait: return "tick-wait";
		case Kind::PriorityWait: return "priority-wait";
		case Kind::BlockedPoll: return "blocked-poll";
		case Kind::Readback: return "readback";
		case Kind::ReadbackClean: return "readback-clean";
		case Kind::DccMetaWrite: return "dcc-meta-write";
		case Kind::DccCheck: return "dcc-check";
		case Kind::DccGpuCheck: return "dcc-gpu-check";
		case Kind::Submit: return "submit";
		case Kind::QueueLockWait: return "queue-lock-wait";
		case Kind::IndirectArgsCpu: return "indirect-cpu";
		case Kind::IndirectArgsGpu: return "indirect-gpu";
		case Kind::GpuBusy: return "gpu-busy";
		case Kind::GpuGap: return "gpu-gap";
		case Kind::OcclusionQuery: return "occlusion-query";
		case Kind::OcclusionPredicate: return "occlusion-pred";
		case Kind::GpuTimestamp: return "gpu-timestamp";
		case Kind::ShaderCompile: return "shader-compile";
		case Kind::PipelineCreate: return "pipeline-create";
		case Kind::Count: break;
	}
	return "?";
}

const char* ReasonName(Reason reason) {
	switch (reason) {
		case Reason::Unattributed: return "unattributed";
		case Reason::GuestReadFault: return "guest-read-fault";
		case Reason::GuestWriteFault: return "guest-write-fault";
		case Reason::GpuThreadReadFault: return "gpu-thread-read-fault";
		case Reason::GpuThreadWriteFault: return "gpu-thread-write-fault";
		case Reason::KernelInvalidate: return "kernel-invalidate";
		case Reason::DccClear: return "dcc-clear";
		case Reason::Predicate: return "predicate";
		case Reason::GdsReadback: return "gds-readback";
		case Reason::Unmap: return "unmap";
		case Reason::TexturePendingDownload: return "texture-pending-download";
		case Reason::FreeImage: return "free-image";
		case Reason::TextureGc: return "texture-gc";
		case Reason::BufferGc: return "buffer-gc";
		case Reason::UploadRingWrap: return "upload-ring-wrap";
		case Reason::StreamRingWrap: return "stream-ring-wrap";
		case Reason::DownloadRingWrap: return "download-ring-wrap";
		case Reason::FaultBuffer: return "fault-buffer";
		case Reason::PresentFrame: return "present-frame";
		case Reason::EagerReadback: return "eager-readback";
		case Reason::Count: break;
	}
	return "?";
}

const char* ZoneName(Zone zone) {
	switch (zone) {
		case Zone::Unmarked: return "unmarked";
		case Zone::GameDraw: return "game-draw";
		case Zone::GameDrawPredicated: return "game-draw-pred";
		case Zone::GameDispatch: return "game-dispatch";
		case Zone::MeshArgs: return "mesh-args";
		case Zone::Tiler: return "tiler";
		case Zone::DccClear: return "dcc-clear";
		case Zone::FaultBuffer: return "fault-buffer";
		case Zone::Blit: return "blit";
		case Zone::ImageCopy: return "image-copy";
		case Zone::BufferCopy: return "buffer-copy";
		case Zone::Count: break;
	}
	return "?";
}

// GPU time per frame by zone, then the costliest zone keys (shader hashes for game work).
std::string ZoneSummary(uint64_t frames) {
	std::map<std::pair<Zone, uint64_t>, ZoneCell> zones;
	{
		std::lock_guard lock(g_zone_mutex);
		zones.swap(g_zones);
	}
	if (zones.empty() || frames == 0) {
		return {};
	}
	const auto per_frame = [frames](uint64_t ns) {
		return static_cast<double>(ns) / 1e6 / static_cast<double>(frames);
	};
	std::array<ZoneCell, static_cast<size_t>(Zone::Count)> totals {};
	uint64_t                                               all_ns = 0;
	for (const auto& [key, cell]: zones) {
		auto& total = totals[static_cast<size_t>(key.first)];
		total.count += cell.count;
		total.ns += cell.ns;
		all_ns += cell.ns;
	}
	std::string text = fmt::format("  gpu-zones      {:.2f}ms/frame:", per_frame(all_ns));
	for (size_t zone = 0; zone < totals.size(); zone++) {
		if (totals[zone].count != 0) {
			text += fmt::format(" {}={:.2f}ms/{:.0f}", ZoneName(static_cast<Zone>(zone)),
			                    per_frame(totals[zone].ns),
			                    static_cast<double>(totals[zone].count) /
			                        static_cast<double>(frames));
		}
	}
	text += '\n';
	std::vector<std::pair<std::pair<Zone, uint64_t>, ZoneCell>> rows(zones.begin(), zones.end());
	std::sort(rows.begin(), rows.end(),
	          [](const auto& a, const auto& b) { return a.second.ns > b.second.ns; });
	for (size_t i = 0; i < std::min<size_t>(rows.size(), 24); i++) {
		const auto& [key, cell] = rows[i];
		text += fmt::format("  gpu-zone       {:<14} key={:016x} {:6.3f}ms/frame n/frame={:.1f} "
		                    "avg={:.1f}us",
		                    ZoneName(key.first), key.second, per_frame(cell.ns),
		                    static_cast<double>(cell.count) / static_cast<double>(frames),
		                    static_cast<double>(cell.ns) / 1e3 / static_cast<double>(cell.count));
		// Draws: the render area per run and the cost per million pixels of it, which stay
		// comparable across the resolutions the game's dynamic scaling picks.
		if (cell.pixels != 0) {
			const auto mpx = static_cast<double>(cell.pixels) / 1e6;
			text += fmt::format(" area={:.2f}Mpx {:.1f}us/Mpx", mpx / static_cast<double>(cell.count),
			                    static_cast<double>(cell.ns) / 1e3 / mpx);
		}
		text += '\n';
	}
	return text;
}

// Percentiles of the time between new game frames, and frames that took an extra refresh.
std::string FrameTimeSummary(const Snapshot& before, const Snapshot& after) {
	std::array<uint64_t, FrameBuckets> counts {};
	uint64_t                           total = 0;
	for (size_t i = 0; i < FrameBuckets; i++) {
		counts[i] = after.frame_ms[i] - before.frame_ms[i];
		total += counts[i];
	}
	if (total == 0) {
		return {};
	}
	const auto percentile = [&](double p) {
		const auto target = static_cast<uint64_t>(p * static_cast<double>(total - 1)) + 1;
		uint64_t   seen   = 0;
		for (size_t i = 0; i < FrameBuckets; i++) {
			seen += counts[i];
			if (seen >= target) {
				return i;
			}
		}
		return FrameBuckets - 1;
	};
	uint64_t long_frames = 0;
	size_t   worst       = 0;
	for (size_t i = 0; i < FrameBuckets; i++) {
		if (counts[i] != 0) {
			worst = i;
		}
		if (i >= 25) {
			long_frames += counts[i];
		}
	}
	return fmt::format("  frame-times    p50={}ms p95={}ms p99={}ms max={}ms >=25ms={} of {} ({:.1f}%)\n",
	                   percentile(0.5), percentile(0.95), percentile(0.99), worst, long_frames, total,
	                   100.0 * static_cast<double>(long_frames) / static_cast<double>(total));
}

std::string OpName(uint32_t op) {
	if (op >= NoPm4Op) {
		return "-";
	}
	if (op >= 256) {
		switch (op - 256) {
			case 0x05: return "R_DRAW_RESET";
			case 0x06: return "R_WAIT_FLIP_DONE";
			case 0x09: return "R_DISPATCH_RESET";
			case 0x14: return "R_ACQUIRE_MEM";
			case 0x15: return "R_WRITE_DATA";
			case 0x17: return "R_FLIP";
			case 0x18: return "R_RELEASE_MEM";
			case 0x19: return "R_DMA_DATA";
			case 0x1A: return "R_CONTEXT_STATE";
			default: return fmt::format("NOP_R{:#04x}", op - 256);
		}
	}
	switch (op) {
		case 0x15: return "DISPATCH_DIRECT";
		case 0x16: return "DISPATCH_INDIRECT";
		case 0x20: return "SET_PREDICATION";
		case 0x22: return "COND_EXEC";
		case 0x24: return "DRAW_INDIRECT";
		case 0x25: return "DRAW_INDEX_INDIRECT";
		case 0x27: return "DRAW_INDEX_2";
		case 0x2C: return "DRAW_INDIRECT_MULTI";
		case 0x2D: return "DRAW_INDEX_AUTO";
		case 0x35: return "DRAW_INDEX_OFFSET_2";
		case 0x37: return "WRITE_DATA";
		case 0x38: return "DRAW_INDEX_INDIRECT_MULTI";
		case 0x3C: return "WAIT_REG_MEM";
		case 0x40: return "COPY_DATA";
		case 0x41: return "CP_DMA";
		case 0x46: return "EVENT_WRITE";
		case 0x47: return "EVENT_WRITE_EOP";
		case 0x48: return "EVENT_WRITE_EOS";
		case 0x49: return "RELEASE_MEM";
		case 0x50: return "DMA_DATA";
		case 0x58: return "ACQUIRE_MEM";
		case 0x8D: return "DISPATCH_DRAW";
		default: return fmt::format("op{:#04x}", op);
	}
}

Snapshot Take() {
	Snapshot snapshot;
	for (size_t i = 0; i < CellCount; i++) {
		snapshot.count[i] = g_cells[i].count.load(std::memory_order_relaxed);
		snapshot.value[i] = g_cells[i].value.load(std::memory_order_relaxed);
	}
	snapshot.frames   = g_frames.load(std::memory_order_relaxed);
	snapshot.presents = g_presents.load(std::memory_order_relaxed);
	for (size_t i = 0; i < FrameBuckets; i++) {
		snapshot.frame_ms[i] = g_frame_ms[i].load(std::memory_order_relaxed);
	}
	return snapshot;
}

struct Row {
	Kind     kind;
	Reason   reason;
	uint32_t op;
	uint64_t count;
	uint64_t value;
};

bool IsTimeKind(Kind kind) {
	switch (kind) {
		case Kind::FullDrain:
		case Kind::TickWait:
		case Kind::PriorityWait:
		case Kind::BlockedPoll:
		case Kind::Submit:
		case Kind::GpuBusy:
		case Kind::GpuGap:
		case Kind::QueueLockWait:
		case Kind::ShaderCompile:
		case Kind::PipelineCreate: return true;
		default: return false;
	}
}

void Report(const Snapshot& before, const Snapshot& after, double seconds, bool interval = true) {
	const auto frames = after.frames - before.frames;
	const auto per    = [frames](double value) { return frames == 0 ? 0.0 : value / frames; };

	std::array<uint64_t, KindCount> kind_count {};
	std::array<uint64_t, KindCount> kind_value {};
	std::vector<Row>                rows;
	for (size_t kind = 0; kind < KindCount; kind++) {
		for (size_t reason = 0; reason < ReasonCount; reason++) {
			for (uint32_t op = 0; op < OpCount; op++) {
				const auto index = Index(static_cast<Kind>(kind), static_cast<Reason>(reason), op);
				const auto count = after.count[index] - before.count[index];
				if (count == 0) {
					continue;
				}
				const auto value = after.value[index] - before.value[index];
				kind_count[kind] += count;
				kind_value[kind] += value;
				rows.push_back({static_cast<Kind>(kind), static_cast<Reason>(reason), op, count,
				                value});
			}
		}
	}

	std::string text = fmt::format(
	    "drain-stats: {:.1f}s frames={} ({:.1f}/s) presents={}", seconds, frames,
	    frames / seconds, after.presents - before.presents);
	for (const auto kind: {Kind::FullDrain, Kind::TickWait, Kind::PriorityWait, Kind::BlockedPoll,
	                       Kind::Submit, Kind::QueueLockWait, Kind::GpuBusy, Kind::GpuGap}) {
		const auto k  = static_cast<size_t>(kind);
		const auto ms = static_cast<double>(kind_value[k]) / 1e6;
		text += fmt::format(" | {} n={} {:.1f}ms ({:.2f}ms/frame)", KindName(kind), kind_count[k],
		                    ms, per(ms));
	}
	const auto rb = static_cast<size_t>(Kind::Readback);
	text += fmt::format(" | readback n={} {:.1f}MiB clean={}\n", kind_count[rb],
	                    static_cast<double>(kind_value[rb]) / (1024.0 * 1024.0),
	                    kind_count[static_cast<size_t>(Kind::ReadbackClean)]);
	text += FrameTimeSummary(before, after);
	// Zones are swapped out per interval, so the session total leaves them out.
	if (interval) {
		text += ZoneSummary(frames);
	}

	std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) {
		const bool a_time = IsTimeKind(a.kind);
		const bool b_time = IsTimeKind(b.kind);
		if (a_time != b_time) {
			return a_time;
		}
		return a_time ? a.value > b.value : a.count > b.count;
	});
	size_t printed = 0;
	for (const auto& row: rows) {
		if (printed++ == 48) {
			break;
		}
		if (IsTimeKind(row.kind)) {
			const auto ms = static_cast<double>(row.value) / 1e6;
			text += fmt::format("  {:<14} {:<26} {:<26} n={:<6} {:8.2f}ms avg={:.3f}ms\n",
			                    KindName(row.kind), ReasonName(row.reason), OpName(row.op),
			                    row.count, ms, ms / static_cast<double>(row.count));
		} else if (row.kind == Kind::OcclusionQuery || row.kind == Kind::OcclusionPredicate ||
		           row.kind == Kind::GpuTimestamp) {
			text += fmt::format("  {:<14} {:<26} {:<26} n={:<6} ({:.1f}/frame)\n", KindName(row.kind),
			                    ReasonName(row.reason), OpName(row.op), row.count,
			                    per(static_cast<double>(row.count)));
		} else if (row.kind == Kind::IndirectArgsCpu || row.kind == Kind::IndirectArgsGpu) {
			text += fmt::format("  {:<14} {:<26} {:<26} n={:<6} mesh={}\n", KindName(row.kind),
			                    ReasonName(row.reason), OpName(row.op), row.count, row.value);
		} else if (row.kind == Kind::DccCheck || row.kind == Kind::DccGpuCheck) {
			text += fmt::format("  {:<14} {:<26} {:<26} n={:<6} {}={}\n", KindName(row.kind),
			                    ReasonName(row.reason), OpName(row.op), row.count,
			                    row.kind == Kind::DccCheck ? "cleared-slices" : "slices",
			                    row.value);
		} else {
			text += fmt::format("  {:<14} {:<26} {:<26} n={:<6} {:8.2f}MiB\n", KindName(row.kind),
			                    ReasonName(row.reason), OpName(row.op), row.count,
			                    static_cast<double>(row.value) / (1024.0 * 1024.0));
		}
	}
	std::unordered_map<uint64_t, FaultSite> sites;
	{
		std::lock_guard lock(g_site_mutex);
		sites.swap(g_sites);
	}
	std::vector<const FaultSite*> site_rows;
	std::vector<uint64_t>         site_keys;
	for (const auto& [key, site]: sites) {
		site_rows.push_back(&site);
		site_keys.push_back(key);
	}
	std::vector<size_t> site_order(site_rows.size());
	for (size_t i = 0; i < site_order.size(); i++) site_order[i] = i;
	std::sort(site_order.begin(), site_order.end(),
	          [&](size_t a, size_t b) { return site_rows[a]->ns > site_rows[b]->ns; });
	for (size_t i = 0; i < std::min<size_t>(site_order.size(), 8); i++) {
		const auto& site = *site_rows[site_order[i]];
		const auto  pc   = site_keys[site_order[i]] & ~(uint64_t {1} << 63u);
		const auto  ms   = static_cast<double>(site.ns) / 1e6;
		text += fmt::format("  fault-site     {:<5} pc={:#014x} thread={:<24} n={:<6} {:8.2f}ms "
		                    "avg={:.3f}ms addr={:#014x}",
		                    site.write ? "write" : "read", pc, site.thread, site.count, ms,
		                    ms / static_cast<double>(site.count), site.last_address);
		if (site.written != 0) {
			text += fmt::format(" writer={} range={:#014x}+{:#x} known={}/{} age={:.2f}frames",
			                    OpName(site.writer_op), site.writer_begin, site.writer_size,
			                    site.written, site.count,
			                    static_cast<double>(site.frame_age) /
			                        static_cast<double>(site.written));
		}
		text += '\n';
	}
	Log::WriteToConsoleAndLog(text);
}

void Run(std::stop_token stop, uint32_t interval_seconds) {
	auto previous      = Take();
	auto previous_time = std::chrono::steady_clock::now();
	const auto first   = previous;
	const auto start   = previous_time;
	while (!stop.stop_requested()) {
		{
			std::unique_lock lock(g_reporter_mutex);
			g_reporter_wake.wait_for(lock, stop, std::chrono::seconds(interval_seconds),
			                         [] { return false; });
		}
		auto current      = Take();
		auto current_time = std::chrono::steady_clock::now();
		if (stop.stop_requested()) {
			if (g_final_report.load(std::memory_order_acquire)) {
				Log::WriteToConsoleAndLog("drain-stats: session total\n");
				Report(first, current,
				       std::chrono::duration<double>(current_time - start).count(), false);
			}
			break;
		}
		Report(previous, current,
		       std::chrono::duration<double>(current_time - previous_time).count());
		previous      = std::move(current);
		previous_time = current_time;
	}
}

} // namespace

void Start(uint32_t interval_seconds) {
	if (interval_seconds == 0 || g_reporter.joinable()) {
		return;
	}
	g_enabled.store(true, std::memory_order_relaxed);
	g_reporter = std::jthread(Run, interval_seconds);
}

void Stop() {
	if (g_reporter.joinable()) {
		g_final_report.store(true, std::memory_order_release);
		g_reporter.request_stop();
		g_reporter.join();
	}
	g_enabled.store(false, std::memory_order_relaxed);
}

void Record(Kind kind, Reason reason, uint32_t pm4_op, uint64_t value) noexcept {
	auto& cell = g_cells[Index(kind, reason, pm4_op)];
	cell.count.fetch_add(1, std::memory_order_relaxed);
	cell.value.fetch_add(value, std::memory_order_relaxed);
	auto& frame = g_frame_cells[static_cast<size_t>(kind)];
	frame.count.fetch_add(1, std::memory_order_relaxed);
	frame.value.fetch_add(value, std::memory_order_relaxed);
}

void RecordFaultSite(uint64_t pc, uint64_t address, bool write, uint64_t ns) noexcept {
	if (!Enabled()) {
		return;
	}
	const auto key = (pc & ~(uint64_t {1} << 63u)) | (write ? uint64_t {1} << 63u : 0);
	GpuWrite   writer;
	if (!write) {
		std::lock_guard lock(g_write_mutex);
		for (size_t i = 1; i <= WriteHistory; i++) {
			const auto& entry = g_writes[(g_write_next + WriteHistory - i) % WriteHistory];
			if (entry.end != 0 && address >= entry.begin && address < entry.end) {
				writer = entry;
				break;
			}
		}
	}
	std::lock_guard lock(g_site_mutex);
	auto&           site = g_sites[key];
	if (writer.end != 0) {
		site.written++;
		site.frame_age += g_frames.load(std::memory_order_relaxed) - writer.frame;
		site.writer_op    = writer.op;
		site.writer_begin = writer.begin;
		site.writer_size  = writer.end - writer.begin;
	}
	if (site.count == 0) {
		char name[64] = "(host thread)";
		if (auto self = LibKernel::PthreadSelfOrNull(); self != nullptr) {
			if (LibKernel::PthreadGetname(self, name) != 0) {
				std::snprintf(name, sizeof(name), "(unnamed guest)");
			}
		}
		site.thread = name;
		site.write  = write;
	}
	site.count++;
	site.ns += ns;
	site.last_address = address;
}

void RecordGpuWrite(uint64_t vaddr, uint64_t size) noexcept {
	if (!Enabled()) {
		return;
	}
	std::lock_guard lock(g_write_mutex);
	g_writes[g_write_next] = {vaddr, vaddr + size, g_frames.load(std::memory_order_relaxed),
	                          t_pm4_op};
	g_write_next = (g_write_next + 1) % WriteHistory;
}

void RecordZones(const ZoneSample* samples, size_t count) noexcept {
	if (!Enabled() || count == 0) {
		return;
	}
	std::lock_guard lock(g_zone_mutex);
	for (size_t i = 0; i < count; i++) {
		auto& cell = g_zones[{samples[i].zone, samples[i].key}];
		cell.count++;
		cell.ns += samples[i].ns;
		cell.pixels += samples[i].pixels;
	}
}

namespace {

// A new game frame took `ms`. Frames of 50 ms or more (a hitch of three or more refreshes) print
// what was recorded during them; every frame then starts a new accumulation.
void ReportHitch(int64_t ms) noexcept {
	constexpr int64_t HitchMs       = 50;
	constexpr size_t  MaxHitchLines = 400;
	static size_t     printed       = 0;
	std::array<uint64_t, KindCount> counts {};
	std::array<uint64_t, KindCount> values {};
	for (size_t kind = 0; kind < KindCount; kind++) {
		counts[kind] = g_frame_cells[kind].count.exchange(0, std::memory_order_relaxed);
		values[kind] = g_frame_cells[kind].value.exchange(0, std::memory_order_relaxed);
	}
	if (ms < HitchMs || printed >= MaxHitchLines) {
		return;
	}
	printed++;
	std::string text = fmt::format("hitch: frame {} took {} ms:", g_frames.load(), ms);
	for (size_t kind = 0; kind < KindCount; kind++) {
		if (counts[kind] == 0) {
			continue;
		}
		const auto k = static_cast<Kind>(kind);
		if (IsTimeKind(k)) {
			text += fmt::format(" {} n={} {:.1f}ms", KindName(k), counts[kind],
			                    static_cast<double>(values[kind]) / 1e6);
		} else if (k == Kind::Readback) {
			text += fmt::format(" {} n={} {:.1f}MiB", KindName(k), counts[kind],
			                    static_cast<double>(values[kind]) / (1024.0 * 1024.0));
		}
	}
	text += '\n';
	Log::WriteToConsoleAndLog(text);
}

} // namespace

void CountFrame(bool new_frame) noexcept {
	if (!Enabled()) {
		return;
	}
	g_presents.fetch_add(1, std::memory_order_relaxed);
	if (new_frame) {
		g_frames.fetch_add(1, std::memory_order_relaxed);
		// Presents come from one thread; time between new game frames, in whole milliseconds.
		static std::chrono::steady_clock::time_point last;
		const auto now = std::chrono::steady_clock::now();
		if (last != std::chrono::steady_clock::time_point {}) {
			const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last).count();
			g_frame_ms[static_cast<size_t>(std::clamp<int64_t>(ms, 0, FrameBuckets - 1))]
			    .fetch_add(1, std::memory_order_relaxed);
			ReportHitch(ms);
		}
		last = now;
	}
}

} // namespace Libs::Graphics::DrainStats
