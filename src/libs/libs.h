#ifndef EMULATOR_INCLUDE_EMULATOR_LIBS_LIBS_H_
#define EMULATOR_INCLUDE_EMULATOR_LIBS_LIBS_H_

#include "common/abi.h"
#include "common/logging/log.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "loader/timer.h" // IWYU pragma: keep

#include <chrono>
#include <cstdlib>
#include <fmt/format.h>

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define PRINT_NAME_ENABLED g_print_name

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define PRINT_NAME_ENABLE(flag) PRINT_NAME_ENABLED = flag;

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LIB_DEFINE(name) void name(Loader::SymbolDatabase* s)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LIB_NAME(l, m)                                                                             \
	[[maybe_unused]] static thread_local bool PRINT_NAME_ENABLED = false;                          \
	static constexpr char                     g_library[]        = l;                              \
	static constexpr char                     g_module[]         = m;
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LIB_VERSION(l, lv, m, mv1, mv2)                                                            \
	LIB_NAME(l, m);                                                                                \
	static constexpr int g_library_version      = lv;                                              \
	static constexpr int g_module_version_major = mv1;                                             \
	static constexpr int g_module_version_minor = mv2;
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LIB_ADD(n, f, t)                                                                           \
	{                                                                                              \
		Loader::SymbolResolve sr {};                                                               \
		sr.name                 = n;                                                               \
		sr.library              = g_library;                                                       \
		sr.library_version      = g_library_version;                                               \
		sr.module               = g_module;                                                        \
		sr.module_version_major = g_module_version_major;                                          \
		sr.module_version_minor = g_module_version_minor;                                          \
		sr.type                 = t;                                                               \
		auto        func        = reinterpret_cast<uint64_t>(f);                                   \
		const char* dbg_name    = "" #f;                                                           \
		s->Add(sr, func, dbg_name);                                                                \
	}
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LIB_OBJECT(n, f) LIB_ADD(n, f, Loader::SymbolType::Object)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LIB_FUNC(n, f) LIB_ADD(n, f, Loader::SymbolType::Func)

// The address of the function's return address on the stack, for KYTY_DEBUG_CALL_COUNTS traces.
#if defined(_MSC_VER)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define KYTY_RETURN_ADDRESS_SLOT() _AddressOfReturnAddress()
#else
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define KYTY_RETURN_ADDRESS_SLOT() __builtin_frame_address(0)
#endif

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define PRINT_NAME()                                                                               \
	do {                                                                                           \
		if (PRINT_NAME_ENABLED) {                                                                  \
			Libs::PrintName(g_library, g_module, __func__);                                        \
		}                                                                                          \
		if (Libs::g_count_calls) [[unlikely]] {                                                    \
			Libs::CountCall(g_library, __func__, KYTY_RETURN_ADDRESS_SLOT());                      \
		}                                                                                          \
	} while (false)

namespace Loader {
class SymbolDatabase;
} // namespace Loader

namespace Libs {

// KYTY_DEBUG_CALL_COUNTS=1 counts the calls of every function that uses PRINT_NAME, and a helper
// thread prints each 5 s window's counts, to see what a stalled game keeps calling (or stopped
// calling). TraceCalls(n) also prints the calling thread's next n calls, each with the guest code
// addresses on its stack above the return address, and TraceAllCalls(duration) every thread's
// calls outside the command buffer builders for that long. Otherwise a call pays one flag test.
inline const bool g_count_calls = std::getenv("KYTY_DEBUG_CALL_COUNTS") != nullptr;
[[gnu::noinline]] void CountCall(const char* library, const char* function,
                                 void* return_slot) noexcept;
void TraceCalls(uint32_t count) noexcept;
void TraceAllCalls(std::chrono::milliseconds duration) noexcept;

// Keep the formatting path from inflating fiber functions' stack frames under LTO.
[[gnu::noinline]] inline void PrintName(const char* library, const char* module, const char* function) {
	if (Log::GetDirection() != Log::Direction::Silent) {
		const auto elapsed_ms      = static_cast<uint64_t>(Loader::Timer::GetTimeMs());
		const auto print_name_time = fmt::format(
		    "{:02}:{:02}:{:02}.{:03}", elapsed_ms / 3600000, (elapsed_ms / 60000) % 60,
		    (elapsed_ms / 1000) % 60, elapsed_ms % 1000);
		LOGF_COLOR(Log::Color::Cyan, "[%d][%s] %s::%s::%s()\n",
		           Common::Thread::GetThreadIdUnique(), print_name_time.c_str(), library, module,
		           function);
	}
}

void InitAll(Loader::SymbolDatabase* s);

} // namespace Libs
#endif /* EMULATOR_INCLUDE_EMULATOR_LIBS_LIBS_H_ */
