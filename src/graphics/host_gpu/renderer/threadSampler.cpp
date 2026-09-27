#include "graphics/host_gpu/renderer/threadSampler.h"

#include "common/common.h"

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // IWYU pragma: keep
#include <psapi.h>
#include <tlhelp32.h>
#undef min
#undef max

namespace Libs::Graphics {

namespace {

constexpr size_t   StackCopyBytes = 64 * 1024;
constexpr uint32_t MaxFrames      = 48;
constexpr auto     Window         = std::chrono::seconds(10);

struct Stack {
	std::array<uint64_t, MaxFrames> frames {};
	uint32_t                        count = 0;

	bool operator==(const Stack& other) const {
		return count == other.count &&
		       std::equal(frames.begin(), frames.begin() + count, other.frames.begin());
	}
};

struct StackHash {
	size_t operator()(const Stack& stack) const {
		uint64_t hash = 0xcbf29ce484222325ull;
		for (uint32_t i = 0; i < stack.count; i++) {
			hash = (hash ^ stack.frames[i]) * 0x100000001b3ull;
		}
		return static_cast<size_t>(hash);
	}
};

// Moves every register that points into the sampled stack range into the copy, so that the
// unwinder reads saved registers and return addresses from the copy.
void RebaseRegisters(CONTEXT& context, uint64_t live, uint64_t copy, uint64_t size) {
	for (DWORD64* reg: {&context.Rax, &context.Rcx, &context.Rdx, &context.Rbx, &context.Rsp,
	                    &context.Rbp, &context.Rsi, &context.Rdi, &context.R8, &context.R9,
	                    &context.R10, &context.R11, &context.R12, &context.R13, &context.R14,
	                    &context.R15}) {
		if (*reg >= live && *reg < live + size) {
			*reg = *reg - live + copy;
		}
	}
}

// Unwinds a context whose stack registers point into `copy`. Plain data only: the guarded
// block cannot hold objects with destructors.
uint32_t UnwindCopy(CONTEXT* context, uint64_t live, uint64_t copy, uint64_t size,
                    uint64_t* frames) {
	uint32_t count = 0;
	__try {
		while (count < MaxFrames) {
			frames[count++] = context->Rip;
			DWORD64 image_base = 0;
			auto*   function   = RtlLookupFunctionEntry(context->Rip, &image_base, nullptr);
			if (function == nullptr) {
				// A leaf function: the return address is at the stack pointer.
				if (context->Rsp < copy || context->Rsp + 8 > copy + size) {
					break;
				}
				context->Rip = *reinterpret_cast<const DWORD64*>(context->Rsp);
				context->Rsp += 8;
			} else {
				PVOID   handler_data = nullptr;
				DWORD64 establisher  = 0;
				RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, context->Rip, function, context,
				                 &handler_data, &establisher, nullptr);
			}
			// Restored registers hold live stack addresses.
			RebaseRegisters(*context, live, copy, size);
			if (context->Rip == 0 || context->Rsp < copy || context->Rsp >= copy + size) {
				break;
			}
		}
	} __except (EXCEPTION_EXECUTE_HANDLER) {
		// A corrupt frame ends the stack; the frames before it stay.
	}
	return count;
}

class ThreadSampler {
public:
	ThreadSampler(std::string name, HANDLE thread, uint64_t stack_low, uint64_t stack_high,
	              uint32_t period_us)
	    : m_name(std::move(name)), m_thread(thread), m_stack_low(stack_low),
	      m_stack_high(stack_high), m_period_us(period_us),
	      m_copy(std::make_unique<uint8_t[]>(StackCopyBytes)) {}

	void Run() {
		HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr,
		                                      CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
		auto window_start = std::chrono::steady_clock::now();
		for (;;) {
			if (timer != nullptr) {
				LARGE_INTEGER due {};
				due.QuadPart = -static_cast<LONGLONG>(m_period_us) * 10;
				SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE);
				WaitForSingleObject(timer, INFINITE);
			} else {
				Sleep(1);
			}
			Stack stack;
			if (Sample(stack)) {
				m_stacks[stack]++;
				m_samples++;
			} else {
				m_lost++;
			}
			const auto now = std::chrono::steady_clock::now();
			if (now - window_start >= Window) {
				Dump(std::chrono::duration<double>(now - window_start).count());
				window_start = now;
			}
		}
	}

private:
	bool Sample(Stack& stack) {
		CONTEXT context {};
		context.ContextFlags = CONTEXT_FULL;
		if (SuspendThread(m_thread) == static_cast<DWORD>(-1)) {
			return false;
		}
		// GetThreadContext also waits until the thread has actually stopped.
		const bool got  = GetThreadContext(m_thread, &context) != 0;
		const auto live = static_cast<uint64_t>(context.Rsp);
		uint64_t   size = 0;
		if (got && live >= m_stack_low && live < m_stack_high) {
			size = std::min<uint64_t>(m_stack_high - live, StackCopyBytes);
			std::memcpy(m_copy.get(), reinterpret_cast<const void*>(live), size);
		}
		ResumeThread(m_thread);
		if (size == 0) {
			return false;
		}
		const auto copy = reinterpret_cast<uint64_t>(m_copy.get());
		RebaseRegisters(context, live, copy, size);
		stack.count = UnwindCopy(&context, live, copy, size, stack.frames.data());
		return stack.count != 0;
	}

public:
	struct Module {
		uint64_t    base = 0;
		uint64_t    size = 0;
		std::string name;
	};

	static std::vector<Module> Modules() {
		std::vector<HMODULE> handles(1024);
		DWORD                needed = 0;
		const auto           process = GetCurrentProcess();
		if (!K32EnumProcessModules(process, handles.data(),
		                           static_cast<DWORD>(handles.size() * sizeof(HMODULE)), &needed)) {
			return {};
		}
		handles.resize(std::min<size_t>(handles.size(), needed / sizeof(HMODULE)));
		std::vector<Module> modules;
		for (const auto handle: handles) {
			MODULEINFO info {};
			char       name[MAX_PATH] {};
			if (K32GetModuleInformation(process, handle, &info, sizeof(info)) &&
			    K32GetModuleBaseNameA(process, handle, name, MAX_PATH) != 0) {
				modules.push_back({reinterpret_cast<uint64_t>(info.lpBaseOfDll), info.SizeOfImage,
				                   name});
			}
		}
		return modules;
	}

private:
	void Dump(double seconds) {
		const auto modules = Modules();
		const auto frame   = [&](uint64_t address) {
			for (const auto& module: modules) {
				if (address >= module.base && address < module.base + module.size) {
					char text[MAX_PATH + 32];
					std::snprintf(text, sizeof(text), "%s+0x%llx", module.name.c_str(),
					              static_cast<unsigned long long>(address - module.base));
					return std::string(text);
				}
			}
			char text[32];
			std::snprintf(text, sizeof(text), "?+0x%llx", static_cast<unsigned long long>(address));
			return std::string(text);
		};
		std::vector<std::pair<const Stack*, uint32_t>> sorted;
		sorted.reserve(m_stacks.size());
		for (const auto& [stack, count]: m_stacks) {
			sorted.emplace_back(&stack, count);
		}
		std::ranges::sort(sorted, [](const auto& a, const auto& b) { return a.second > b.second; });
		char path[MAX_PATH];
		std::snprintf(path, sizeof(path), "sample-%s-%03u.txt", m_name.c_str(), m_window);
		if (FILE* file = std::fopen(path, "w"); file != nullptr) {
			std::fprintf(file, "# window=%u seconds=%.2f samples=%llu lost=%llu period_us=%u\n",
			             m_window, seconds, static_cast<unsigned long long>(m_samples),
			             static_cast<unsigned long long>(m_lost), m_period_us);
			for (const auto& [stack, count]: sorted) {
				std::fprintf(file, "%u", count);
				for (uint32_t i = 0; i < stack->count; i++) {
					std::fprintf(file, "%c%s", i == 0 ? ' ' : ';', frame(stack->frames[i]).c_str());
				}
				std::fputc('\n', file);
			}
			std::fclose(file);
		}
		m_window++;
		m_stacks.clear();
		m_samples = 0;
		m_lost    = 0;
	}

	std::string                                  m_name;
	HANDLE                                       m_thread;
	uint64_t                                     m_stack_low;
	uint64_t                                     m_stack_high;
	uint32_t                                     m_period_us;
	std::unique_ptr<uint8_t[]>                   m_copy;
	std::unordered_map<Stack, uint32_t, StackHash> m_stacks;
	uint64_t                                     m_samples = 0;
	uint64_t                                     m_lost    = 0;
	uint32_t                                     m_window  = 0;
};

// Writes the call stack of every other thread of the process to thread-dump-<index>.txt: a
// "# tid=<id> name=<description>" line, then "1 <frames>" in the sampler's format. Guest code has
// no unwind data, so a stack usually ends at the first guest frame after the emulator's own.
void DumpThreads(uint32_t index) {
	const auto process = GetCurrentProcessId();
	const auto self    = GetCurrentThreadId();
	std::vector<DWORD> ids;
	if (HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
	    snapshot != INVALID_HANDLE_VALUE) {
		THREADENTRY32 entry {};
		entry.dwSize = sizeof(entry);
		for (BOOL more = Thread32First(snapshot, &entry); more;
		     more      = Thread32Next(snapshot, &entry)) {
			if (entry.th32OwnerProcessID == process && entry.th32ThreadID != self) {
				ids.push_back(entry.th32ThreadID);
			}
		}
		CloseHandle(snapshot);
	}
	char path[MAX_PATH];
	std::snprintf(path, sizeof(path), "thread-dump-%03u.txt", index);
	FILE* file = std::fopen(path, "w");
	if (file == nullptr) {
		return;
	}
	std::fprintf(file, "# window=%u seconds=0 samples=%zu lost=0 period_us=0\n", index, ids.size());
	auto       copy    = std::make_unique<uint8_t[]>(StackCopyBytes);
	const auto modules = ThreadSampler::Modules();
	for (const auto id: ids) {
		HANDLE thread = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT |
		                               THREAD_QUERY_LIMITED_INFORMATION,
		                           FALSE, id);
		if (thread == nullptr) {
			continue;
		}
		std::string name;
		PWSTR       description = nullptr;
		if (SUCCEEDED(GetThreadDescription(thread, &description)) && description != nullptr) {
			for (const auto* c = description; *c != 0; c++) {
				name.push_back(*c < 128 ? static_cast<char>(*c) : '?');
			}
			LocalFree(description);
		}
		// Nothing may allocate or lock while the thread is suspended: it may hold those locks.
		CONTEXT context {};
		context.ContextFlags = CONTEXT_FULL;
		uint64_t live        = 0;
		uint64_t size        = 0;
		if (SuspendThread(thread) != static_cast<DWORD>(-1)) {
			if (GetThreadContext(thread, &context) != 0) {
				live = context.Rsp;
				MEMORY_BASIC_INFORMATION region {};
				if (VirtualQuery(reinterpret_cast<const void*>(live), &region, sizeof(region)) != 0 &&
				    region.State == MEM_COMMIT) {
					const auto end = reinterpret_cast<uint64_t>(region.BaseAddress) + region.RegionSize;
					size           = std::min<uint64_t>(end - live, StackCopyBytes);
					std::memcpy(copy.get(), reinterpret_cast<const void*>(live), size);
				}
			}
			ResumeThread(thread);
		}
		CloseHandle(thread);
		std::fprintf(file, "# tid=%lu name=%s\n", id, name.empty() ? "-" : name.c_str());
		if (size == 0) {
			continue;
		}
		Stack      stack;
		const auto base = reinterpret_cast<uint64_t>(copy.get());
		RebaseRegisters(context, live, base, size);
		stack.count = UnwindCopy(&context, live, base, size, stack.frames.data());
		std::fprintf(file, "1");
		for (uint32_t i = 0; i < stack.count; i++) {
			const auto address = stack.frames[i];
			const ThreadSampler::Module* owner = nullptr;
			for (const auto& module: modules) {
				if (address >= module.base && address < module.base + module.size) {
					owner = &module;
					break;
				}
			}
			if (owner != nullptr) {
				std::fprintf(file, "%c%s+0x%llx", i == 0 ? ' ' : ';', owner->name.c_str(),
				             static_cast<unsigned long long>(address - owner->base));
			} else {
				std::fprintf(file, "%c?+0x%llx", i == 0 ? ' ' : ';',
				             static_cast<unsigned long long>(address));
			}
		}
		std::fputc('\n', file);
	}
	std::fclose(file);
}

} // namespace

void StartThreadSampler(const char* name) {
	const char* value = std::getenv("KYTY_DEBUG_SAMPLE_GPU");
	if (value == nullptr) {
		return;
	}
	auto period_us = static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
	if (period_us < 100) {
		period_us = 500;
	}
	HANDLE thread = nullptr;
	if (!DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &thread,
	                     THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION,
	                     FALSE, 0)) {
		return;
	}
	ULONG_PTR low  = 0;
	ULONG_PTR high = 0;
	GetCurrentThreadStackLimits(&low, &high);
	std::thread([sampler = std::make_shared<ThreadSampler>(name, thread, low, high, period_us)] {
		SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
		sampler->Run();
	}).detach();
	std::printf("thread-sampler: sampling %s every %u us\n", name, period_us);
}

void StartThreadDumper() {
	const char* value = std::getenv("KYTY_DEBUG_DUMP_THREADS");
	if (value == nullptr) {
		return;
	}
	const auto period = std::max(1ul, std::strtoul(value, nullptr, 10));
	std::thread([period] {
		for (uint32_t index = 0;; index++) {
			std::this_thread::sleep_for(std::chrono::seconds(period));
			DumpThreads(index);
		}
	}).detach();
	std::printf("thread-dumper: dumping all threads every %lu s\n", period);
}

} // namespace Libs::Graphics

#else

namespace Libs::Graphics {

void StartThreadSampler(const char* /*name*/) {}
void StartThreadDumper() {}

} // namespace Libs::Graphics

#endif
