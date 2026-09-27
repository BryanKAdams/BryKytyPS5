#include "libs/libs.h"

#include "common/logging/log.h"
#include "common/singleton.h"
#include "libs/errno.h"
#include "loader/runtimeLinker.h"
#include "loader/symbolDatabase.h"

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace Libs {

namespace {

// KYTY_DEBUG_CALL_COUNTS: one slot per function, claimed by its __func__ pointer.
struct CallCounter {
	std::atomic<const char*> function {nullptr};
	std::atomic<const char*> library {nullptr};
	std::atomic<uint64_t>    count {0};
	// KYTY_DEBUG_TRACE_FUNCTIONS: whether this function is traced, and its calls left to trace in
	// the current 5 s window.
	std::atomic<bool>        traced {false};
	std::atomic<int64_t>     trace_budget {0};
};
constexpr size_t                          CallCounterCount = 4096;
std::array<CallCounter, CallCounterCount> g_call_counters;

// KYTY_DEBUG_TRACE_FUNCTIONS=<name>,<name>,...: prints the guest callers of the first
// KYTY_DEBUG_TRACE_LIMIT (default 100) calls of each named function in every 5 s window (names as
// call-counts prints them, without the library).
int64_t TraceBudget(const char* function) {
	static const std::vector<std::string> names = [] {
		std::vector<std::string> list;
		if (const char* value = std::getenv("KYTY_DEBUG_TRACE_FUNCTIONS"); value != nullptr) {
			std::string text = value;
			for (size_t begin = 0; begin <= text.size();) {
				const auto end = std::min(text.find(',', begin), text.size());
				if (end > begin) {
					list.push_back(text.substr(begin, end - begin));
				}
				begin = end + 1;
			}
		}
		return list;
	}();
	static const int64_t limit = [] {
		const char* value = std::getenv("KYTY_DEBUG_TRACE_LIMIT");
		return value != nullptr ? std::max<int64_t>(std::atoll(value), 0) : int64_t {100};
	}();
	return std::find(names.begin(), names.end(), function) != names.end() ? limit : 0;
}

void PrintCallCounts() {
	for (uint64_t window = 1;; window++) {
		std::this_thread::sleep_for(std::chrono::seconds(5));
		std::vector<std::pair<uint64_t, const CallCounter*>> calls;
		uint64_t                                             total = 0;
		for (auto& counter: g_call_counters) {
			const auto count = counter.count.exchange(0);
			if (counter.traced.load(std::memory_order_relaxed)) {
				counter.trace_budget.store(TraceBudget(counter.function.load()));
			}
			if (count != 0) {
				calls.emplace_back(count, &counter);
				total += count;
			}
		}
		std::sort(calls.begin(), calls.end(),
		          [](const auto& a, const auto& b) { return a.first > b.first; });
		std::string line = fmt::format("call-counts: t={}s total={}", window * 5, total);
		for (size_t i = 0; i < std::min<size_t>(calls.size(), 200); i++) {
			const char* library  = calls[i].second->library.load();
			const char* function = calls[i].second->function.load();
			line += fmt::format(" {}::{}={}", library != nullptr ? library : "?",
			                    function != nullptr ? function : "?", calls[i].first);
		}
		std::printf("%s\n", line.c_str());
		std::fflush(stdout);
	}
}

thread_local uint32_t t_trace_calls = 0;

// Guest code addresses among the stack qwords from the return address up, within the stack
// region: the return address, then the callers (and whatever else on the stack points into code).
void PrintTracedCall(const char* library, const char* function, void* return_slot) {
	const auto begin = reinterpret_cast<uint64_t>(return_slot);
	uint64_t   end   = begin + 64 * sizeof(uint64_t);
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	MEMORY_BASIC_INFORMATION info {};
	if (VirtualQuery(return_slot, &info, sizeof(info)) != 0) {
		end = std::min(begin + 256 * sizeof(uint64_t),
		               reinterpret_cast<uint64_t>(info.BaseAddress) + info.RegionSize);
	}
#endif
	auto*       linker = Common::Singleton<Loader::RuntimeLinker>::Instance();
	std::string line   = fmt::format("call-trace tid={} {}::{}", Common::Thread::GetThreadIdUnique(),
	                                 library, function);
	int         found  = 0;
	for (uint64_t slot = begin; slot + sizeof(uint64_t) <= end && found < 12;
	     slot += sizeof(uint64_t)) {
		const auto value   = *reinterpret_cast<const uint64_t*>(slot);
		auto*      program = linker->FindProgramByAddr(value);
		if (program == nullptr) {
			continue;
		}
		line += fmt::format(" {}+0x{:x}@{:x}", Common::PathToString(program->file_name.filename()),
		                    value - program->base_vaddr, slot - begin);
		found++;
	}
	std::printf("%s\n", line.c_str());
	std::fflush(stdout);
}

// End of the all-threads trace window, in steady_clock ticks; 0 when none is open.
std::atomic<int64_t> g_trace_all_until {0};

} // namespace

void TraceCalls(uint32_t count) noexcept {
	t_trace_calls = count;
}

void TraceAllCalls(std::chrono::milliseconds duration) noexcept {
	g_trace_all_until = (std::chrono::steady_clock::now() + duration).time_since_epoch().count();
}

// PRINT_NAME's g_count_call when KYTY_DEBUG_CALL_COUNTS is set.
static void CountCall(const char* library, const char* function, void* return_slot) {
	static std::once_flag printer;
	std::call_once(printer, [] { std::thread(PrintCallCounts).detach(); });
	if (t_trace_calls != 0) [[unlikely]] {
		t_trace_calls--;
		PrintTracedCall(library, function, return_slot);
	} else if (const auto until = g_trace_all_until.load(std::memory_order_relaxed); until != 0)
	    [[unlikely]] {
		// The command buffer builders would drown out the other threads.
		if (std::chrono::steady_clock::now().time_since_epoch().count() >= until) {
			auto expected = until;
			g_trace_all_until.compare_exchange_strong(expected, 0);
		} else if (std::strncmp(library, "Graphics5", 9) != 0) {
			PrintTracedCall(library, function, return_slot);
		}
	}
	const auto first = (reinterpret_cast<uintptr_t>(function) >> 4u) % CallCounterCount;
	for (size_t probe = 0; probe < CallCounterCount; probe++) {
		auto&       counter = g_call_counters[(first + probe) % CallCounterCount];
		const char* owner   = counter.function.load(std::memory_order_acquire);
		if (owner == nullptr) {
			if (counter.function.compare_exchange_strong(owner, function)) {
				counter.library.store(library);
				const auto budget = TraceBudget(function);
				counter.trace_budget.store(budget);
				counter.traced.store(budget > 0);
				owner = function;
			}
		}
		if (owner == function) {
			counter.count.fetch_add(1, std::memory_order_relaxed);
			if (counter.trace_budget.load(std::memory_order_relaxed) > 0 &&
			    counter.trace_budget.fetch_sub(1, std::memory_order_relaxed) > 0) [[unlikely]] {
				PrintTracedCall(library, function, return_slot);
			}
			return;
		}
	}
}

[[maybe_unused]] static const bool g_count_call_installed = [] {
	if (std::getenv("KYTY_DEBUG_CALL_COUNTS") != nullptr) {
		g_count_call = CountCall;
	}
	return true;
}();

namespace LibContentDelete {
LIB_DEFINE(InitContentDelete_1);
} // namespace LibContentDelete

namespace LibContentExport {
LIB_DEFINE(InitContentExport_1);
} // namespace LibContentExport

namespace LibContentSearch {
LIB_DEFINE(InitContentSearch_1);
} // namespace LibContentSearch

namespace VideoDec2 {
LIB_DEFINE(InitVideoDec2_1);
} // namespace VideoDec2

namespace LibMouse {
LIB_DEFINE(InitMouse_1);
} // namespace LibMouse

namespace LibKeyboard {
LIB_DEFINE(InitKeyboard_1);
} // namespace LibKeyboard

namespace Ime {
LIB_DEFINE(InitPlatform_1_Ime);
} // namespace Ime

namespace LibUlt {
LIB_DEFINE(InitUlt_1);
} // namespace LibUlt

namespace LibPsml {
LIB_DEFINE(InitPsml_1);
} // namespace LibPsml

namespace LibCes {
LIB_DEFINE(InitCes_1);
} // namespace LibCes

namespace LibC {
LIB_DEFINE(InitLibC_1);
} // namespace LibC

namespace LibAmpr {
LIB_DEFINE(InitAmpr_1);
} // namespace LibAmpr

namespace Coredump {
LIB_DEFINE(InitCoredump_1);
} // namespace Coredump

namespace LibRazorCpu {
LIB_DEFINE(InitRazorCpu_1);
} // namespace LibRazorCpu

namespace Fiber {
LIB_DEFINE(InitFiber_1);
} // namespace Fiber

namespace LibRtc {
LIB_DEFINE(InitRtc_1);
} // namespace LibRtc

namespace LibGen5 {
LIB_DEFINE(InitVideoOut_1);
namespace VrrStatus {
LIB_DEFINE(InitVideoOutVrrStatus_1);
} // namespace VrrStatus
} // namespace LibGen5

LIB_DEFINE(InitAppContent_1);
LIB_DEFINE(InitAudio_1);
LIB_DEFINE(InitConvertKeycode_1);
LIB_DEFINE(InitDbgAddressSanitizer_1);
LIB_DEFINE(InitDialog_1);
LIB_DEFINE(InitFont_1);
LIB_DEFINE(InitFontFt_1);
LIB_DEFINE(InitAgcDriver_1);
LIB_DEFINE(InitHmd2_1);
LIB_DEFINE(InitLibKernel_1);
LIB_DEFINE(InitNet_1);
LIB_DEFINE(InitPad_1);
LIB_DEFINE(InitPlayGo_1);
LIB_DEFINE(InitPngDec_1);
LIB_DEFINE(InitPlatform_1);
LIB_DEFINE(InitRudp_1);
LIB_DEFINE(InitSaveData_1);
LIB_DEFINE(InitShare_1);
LIB_DEFINE(InitSysmodule_1);
LIB_DEFINE(InitSystemService_1);
LIB_DEFINE(InitTextToSpeech2_1);
LIB_DEFINE(InitUserService_1);
LIB_DEFINE(InitWebBrowserDialog_1);

void InitAll(Loader::SymbolDatabase* s) {
	InitAudio_1(s);
	InitConvertKeycode_1(s);
	LibAmpr::InitAmpr_1(s);
	InitAppContent_1(s);
	Coredump::InitCoredump_1(s);
	LibContentDelete::InitContentDelete_1(s);
	LibContentExport::InitContentExport_1(s);
	LibContentSearch::InitContentSearch_1(s);
	LibC::InitLibC_1(s);
	LibCes::InitCes_1(s);
	InitDbgAddressSanitizer_1(s);
	LibRazorCpu::InitRazorCpu_1(s);
	InitDialog_1(s);
	Fiber::InitFiber_1(s);
	InitFont_1(s);
	InitFontFt_1(s);
	InitAgcDriver_1(s);
	InitHmd2_1(s);
	InitLibKernel_1(s);
	LibMouse::InitMouse_1(s);
	LibKeyboard::InitKeyboard_1(s);
	Ime::InitPlatform_1_Ime(s);
	InitNet_1(s);
	InitPad_1(s);
	InitPlayGo_1(s);
	LibPsml::InitPsml_1(s);
	InitPngDec_1(s);
	InitPlatform_1(s);
	InitRudp_1(s);
	LibRtc::InitRtc_1(s);
	InitSaveData_1(s);
	InitShare_1(s);
	InitSysmodule_1(s);
	InitSystemService_1(s);
	InitTextToSpeech2_1(s);
	LibUlt::InitUlt_1(s);
	InitUserService_1(s);
	VideoDec2::InitVideoDec2_1(s);
	LibGen5::InitVideoOut_1(s);
	LibGen5::VrrStatus::InitVideoOutVrrStatus_1(s);
	InitWebBrowserDialog_1(s);
}

namespace LibContentExport {

LIB_VERSION("ContentExport", 1, "ContentExport", 1, 1);

namespace ContentExport {

constexpr int CONTENT_EXPORT_ERROR_INVALID_PARAM = -2137182186; /* 0x809D3016 */

struct ContentExportInitParam2 {
	void*   malloc_func;
	void*   free_func;
	void*   user_data;
	size_t  buffer_size;
	int64_t reserved0;
	int64_t reserved1;
};

static bool g_initialized = false;

static int KYTY_SYSV_ABI ContentExportInit2(const ContentExportInitParam2* init_param) {
	PRINT_NAME();

	LOGF("\t init_param  = 0x%016" PRIx64 "\n", reinterpret_cast<uint64_t>(init_param));

	if (init_param == nullptr) {
		return CONTENT_EXPORT_ERROR_INVALID_PARAM;
	}

	LOGF("\t malloc_func = 0x%016" PRIx64 "\n"
	     "\t free_func   = 0x%016" PRIx64 "\n"
	     "\t user_data   = 0x%016" PRIx64 "\n"
	     "\t buffer_size = %" PRIu64 "\n",
	     reinterpret_cast<uint64_t>(init_param->malloc_func),
	     reinterpret_cast<uint64_t>(init_param->free_func),
	     reinterpret_cast<uint64_t>(init_param->user_data),
	     static_cast<uint64_t>(init_param->buffer_size));

	g_initialized = true;

	return OK;
}

} // namespace ContentExport

LIB_DEFINE(InitContentExport_1) {
	LIB_FUNC("0GnN4QCgIfs", ContentExport::ContentExportInit2);
}

} // namespace LibContentExport

namespace LibContentSearch {

LIB_VERSION("ContentSearch", 1, "ContentSearch", 1, 0);

namespace ContentSearch {

constexpr int CONTENT_SEARCH_ERROR_INVALID_PARAM = -2137190397; /* 0x809D1003 */

struct ContentSearchInitParam {
	size_t memory_size;
};

static bool g_initialized = false;

static int KYTY_SYSV_ABI ContentSearchInit(const ContentSearchInitParam* init_param) {
	PRINT_NAME();

	LOGF("\t init_param  = 0x%016" PRIx64 "\n", reinterpret_cast<uint64_t>(init_param));

	if (init_param == nullptr) {
		return CONTENT_SEARCH_ERROR_INVALID_PARAM;
	}

	LOGF("\t memory_size = %" PRIu64 "\n", static_cast<uint64_t>(init_param->memory_size));

	g_initialized = true;

	return OK;
}

} // namespace ContentSearch

LIB_DEFINE(InitContentSearch_1) {
	LIB_FUNC("dPj4ZtRcIWk", ContentSearch::ContentSearchInit);
}

} // namespace LibContentSearch

namespace LibContentDelete {

LIB_VERSION("ContentDelete", 1, "ContentDelete", 1, 1);

namespace ContentDelete {

constexpr int CONTENT_DELETE_ERROR_INVALID_PARAM = -2137174015; /* 0x809D5001 */

struct ContentDeleteInitParam {
	char   reserved1[4];
	size_t heap_size;
	char   reserved2[32];
};

static bool g_initialized = false;

static int KYTY_SYSV_ABI ContentDeleteInitialize(const ContentDeleteInitParam* init_param) {
	PRINT_NAME();

	LOGF("\t init_param = 0x%016" PRIx64 "\n", reinterpret_cast<uint64_t>(init_param));

	if (init_param == nullptr) {
		return CONTENT_DELETE_ERROR_INVALID_PARAM;
	}

	LOGF("\t heap_size  = %" PRIu64 "\n", static_cast<uint64_t>(init_param->heap_size));

	g_initialized = true;

	return OK;
}

} // namespace ContentDelete

LIB_DEFINE(InitContentDelete_1) {
	LIB_FUNC("zoxb0wEChEM", ContentDelete::ContentDeleteInitialize);
}

} // namespace LibContentDelete

} // namespace Libs
