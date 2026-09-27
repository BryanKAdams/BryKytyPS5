#include "common/hostException.h"
#include "common/virtualMemory.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/rangeSet.h"

#include <atomic>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <semaphore>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#else
#include <csignal>
#include <map>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

using Libs::Graphics::GuestRange;
using Libs::Graphics::MemoryTracker;
using Libs::Graphics::PageManager;
using Libs::Graphics::RangeSet;
using Libs::Graphics::TRACKER_ADDRESS_SIZE;

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "MemoryTrackerTests: failed: %s\n", text);
    std::abort();
  }
}

#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
using DWORD = uint32_t;
constexpr uint32_t PAGE_NOACCESS = 1;
constexpr uint32_t PAGE_READONLY = 2;
constexpr uint32_t PAGE_READWRITE = 3;
constexpr uint32_t MEM_RESERVE = 0;
constexpr uint32_t MEM_COMMIT = 0;
constexpr uint32_t MEM_RELEASE = 0;

int ToHostProt(uint32_t protection) {
  switch (protection) {
  case PAGE_NOACCESS:
    return PROT_NONE;
  case PAGE_READONLY:
    return PROT_READ;
  default:
    return PROT_READ | PROT_WRITE;
  }
}

uint32_t Protection(const void *address) {
  const auto addr = reinterpret_cast<uintptr_t>(address);
  std::FILE *maps = std::fopen("/proc/self/maps", "r");
  Check(maps != nullptr, "open /proc/self/maps failed");
  char line[512];
  uint32_t result = 0;
  while (std::fgets(line, sizeof(line), maps) != nullptr) {
    unsigned long start = 0;
    unsigned long end = 0;
    char perms[8]{};
    if (std::sscanf(line, "%lx-%lx %7s", &start, &end, perms) != 3) {
      continue;
    }
    if (addr >= start && addr < end) {
      result = perms[1] == 'w'   ? PAGE_READWRITE
               : perms[0] == 'r' ? PAGE_READONLY
                                 : PAGE_NOACCESS;
      break;
    }
  }
  std::fclose(maps);
  return result;
}

std::map<void *, size_t> &AllocationSizes() {
  static std::map<void *, size_t> sizes;
  return sizes;
}

void *VirtualAlloc(void *address, size_t size, DWORD, uint32_t protection) {
  const int extra = address != nullptr ? MAP_FIXED_NOREPLACE : 0;
  void *raw = ::mmap(address, size, ToHostProt(protection),
                     MAP_PRIVATE | MAP_ANONYMOUS | extra, -1, 0);
  if (raw == MAP_FAILED) {
    return nullptr;
  }
  AllocationSizes()[raw] = size;
  return raw;
}

int VirtualFree(void *address, size_t, DWORD) {
  auto &sizes = AllocationSizes();
  auto it = sizes.find(address);
  if (it == sizes.end()) {
    return 0;
  }
  const int ok = ::munmap(address, it->second) == 0 ? 1 : 0;
  sizes.erase(it);
  return ok;
}

int VirtualProtect(void *address, size_t size, uint32_t protection,
                   DWORD *old_protection) {
  if (old_protection != nullptr) {
    *old_protection = Protection(address);
  }
  return ::mprotect(address, size, ToHostProt(protection)) == 0 ? 1 : 0;
}
#else
uint32_t Protection(const void *address) {
  MEMORY_BASIC_INFORMATION info{};
  Check(VirtualQuery(address, &info, sizeof(info)) != 0, "VirtualQuery failed");
  return info.Protect;
}
#endif

bool IsWritable(const void *address) {
  return Protection(address) == PAGE_READWRITE;
}

uint64_t g_protection_calls = 0;

struct ProtectionCall {
  uint64_t address;
  uint64_t size;
  Common::VirtualMemory::Mode mode;
};

std::vector<ProtectionCall> g_protection_log;

void ResetProtectionLog() {
  g_protection_calls = 0;
  g_protection_log.clear();
}

bool ProtectAddressSpace(uint64_t vaddr, uint64_t size,
                         Common::VirtualMemory::Mode mode) {
  uint32_t protection = PAGE_NOACCESS;
  if (mode == Common::VirtualMemory::Mode::Read) {
    protection = PAGE_READONLY;
  } else if (mode == Common::VirtualMemory::Mode::ReadWrite) {
    protection = PAGE_READWRITE;
  }
  DWORD old_protection = 0;
  g_protection_calls++;
  g_protection_log.push_back({vaddr, size, mode});
  return VirtualProtect(reinterpret_cast<void *>(vaddr), size, protection,
                        &old_protection) != 0;
}

struct TrackerHarness {
  TrackerHarness() : tracker(page_manager) {}

  PageManager page_manager;
  MemoryTracker tracker;
};

uint8_t *Allocate(PageManager &manager, uint64_t pages) {
  constexpr uintptr_t base = 0x0000000200010000ull;
  const auto size = manager.GetPageSize() * pages;
  auto *memory = static_cast<uint8_t *>(
      VirtualAlloc(reinterpret_cast<void *>(base), size,
                   MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
  Check(memory == reinterpret_cast<void *>(base), "fixed VirtualAlloc failed");
  return memory;
}

void Release(uint8_t *memory) {
  Check(VirtualFree(memory, 0, MEM_RELEASE) != 0, "VirtualFree failed");
}

void TestRangeSet() {
  RangeSet ranges;
  ranges.Add(0x1000, 0x80);
  ranges.Add(0x1080, 0x80);
  ranges.Add(0x1200, 0x40);
  Check(ranges.Contains(0x1010, 0xe0) && !ranges.Contains(0x1010, 0x200),
        "range set containment did not require full coverage");
  Check(ranges.Intersects(0x0fff, 2) && ranges.Intersects(0x11ff, 2) &&
            !ranges.Intersects(0x1100, 0x100) &&
            !ranges.Intersects(0x1240, 1),
        "range set intersection did not preserve half-open boundaries");
  std::vector<std::pair<uint64_t, uint64_t>> intersections;
  ranges.ForEachInRange(0x1070, 0x1b0, [&](uint64_t start, uint64_t end) {
    intersections.emplace_back(start, end);
  });
  Check(intersections.size() == 2 && intersections[0].first == 0x1070 &&
            intersections[0].second == 0x1100 &&
            intersections[1].first == 0x1200 && intersections[1].second == 0x1220,
        "range set did not merge and intersect exact byte ranges");
  ranges.Subtract(0x1040, 0x1e0);
  intersections.clear();
  ranges.ForEachInRange(0x1000, 0x300, [&](uint64_t start, uint64_t end) {
    intersections.emplace_back(start, end);
  });
  Check(intersections.size() == 2 && intersections[0].first == 0x1000 &&
            intersections[0].second == 0x1040 &&
            intersections[1].first == 0x1220 && intersections[1].second == 0x1240,
        "range set subtraction did not preserve both exact tails");
}

void TestGuestRange() {
  constexpr GuestRange empty{};
  constexpr GuestRange first_byte{1, 1};
  constexpr GuestRange last_byte{TRACKER_ADDRESS_SIZE - 1, 1};

  static_assert(empty.Empty() && !empty.Valid() && empty.ValidOrEmpty());
  static_assert(!first_byte.Empty() && first_byte.Valid() &&
                first_byte.ValidOrEmpty() && first_byte.End() == 2);
  static_assert(last_byte.Valid() && last_byte.End() == TRACKER_ADDRESS_SIZE);

  Check(!GuestRange{0, 1}.Empty() && !GuestRange{0, 1}.ValidOrEmpty(),
        "zero-address nonempty guest range is rejected");
  Check(!GuestRange{1, 0}.Empty() && !GuestRange{1, 0}.ValidOrEmpty(),
        "nonzero-address empty guest range is rejected");
  Check(!GuestRange{TRACKER_ADDRESS_SIZE, 1}.Valid(),
        "first address beyond the guest range is rejected");
  Check(!GuestRange{TRACKER_ADDRESS_SIZE - 1, 2}.Valid(),
        "guest range crossing the address-space end is rejected");
  Check(!GuestRange{UINT64_MAX, 2}.Valid(), "wrapping guest range is rejected");
}

void TestQueriesDoNotRequireMappedOwnership() {
  constexpr uint64_t address = 0x0000000203000000ull;
  TrackerHarness harness;
  const auto page_size = harness.page_manager.GetPageSize();
  Check(harness.tracker.IsRegionCpuModified(address, page_size) &&
            !harness.tracker.IsRegionGpuModified(address, page_size),
        "unowned tracker range did not expose its initial CPU-dirty state");
}

void TestConcurrentRegionPublication() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 1);
  const auto address = reinterpret_cast<uint64_t>(memory);

  std::binary_semaphore start_first{0};
  std::binary_semaphore start_second{0};
  std::atomic_uint32_t cpu_dirty_results{0};
  std::jthread first([&] {
    start_first.acquire();
    if (tracker.IsRegionCpuModified(address, page_size)) {
      cpu_dirty_results.fetch_add(1, std::memory_order_relaxed);
    }
  });
  std::jthread second([&] {
    start_second.acquire();
    if (tracker.IsRegionCpuModified(address, page_size)) {
      cpu_dirty_results.fetch_add(1, std::memory_order_relaxed);
    }
  });
  start_first.release();
  start_second.release();
  first.join();
  second.join();

  tracker.UntrackMemory(address, page_size);
  Release(memory);
  Check(cpu_dirty_results.load(std::memory_order_relaxed) == 2,
        "concurrent region publication lost initial CPU ownership");
}

void TestCpuDirtyUpload() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 2);
  const auto address = reinterpret_cast<uint64_t>(memory);
  Check(tracker.IsRegionCpuModified(address + 16, 32),
        "new region was not CPU dirty");

  uint32_t ranges = 0;
  bool uploaded = false;
  tracker.ForEachUploadRange(
      address + 16, 32, false,
      [&](uint64_t upload_address, uint64_t upload_size) noexcept {
        Check(upload_address == address && upload_size == page_size,
              "upload range was not page aligned");
        ranges++;
      },
      [&]() noexcept { uploaded = true; });
  Check(ranges == 1 && uploaded &&
            !tracker.IsRegionCpuModified(address, page_size) &&
            Protection(memory) == PAGE_READONLY,
        "upload did not clear CPU dirty state and arm protection");

  tracker.MarkRegionAsCpuModified(address + 16, 32);
  Check(tracker.IsRegionCpuModified(address, page_size) && IsWritable(memory),
        "explicit CPU dirtiness did not release write protection");
  tracker.UntrackMemory(address, page_size * 2);
  Release(memory);
}

// IsRegionCpuCleanHint follows the CPU-dirty bits per 64 KiB slice: an untracked region is not
// clean, an uploaded range is, and a CPU write only affects the slice it lands in.
void TestCpuCleanHint() {
  constexpr uintptr_t base = 0x0000000203000000ull;
  constexpr uint64_t slice = uint64_t{64} * 1024;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto *memory = static_cast<uint8_t *>(
      VirtualAlloc(reinterpret_cast<void *>(base), slice * 3,
                   MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
  Check(memory == reinterpret_cast<void *>(base), "fixed VirtualAlloc failed");
  const auto address = reinterpret_cast<uint64_t>(memory);
  const auto upload = [&](uint64_t upload_address, uint64_t upload_size) {
    tracker.ForEachUploadRange(
        upload_address, upload_size, false,
        [](uint64_t, uint64_t) noexcept {}, []() noexcept {});
  };

  Check(!tracker.IsRegionCpuCleanHint(address, 64),
        "an untracked region counted as CPU clean");
  upload(address, slice * 3);
  Check(tracker.IsRegionCpuCleanHint(address, slice * 3),
        "an uploaded range was not CPU clean");
  Check(!tracker.IsRegionCpuCleanHint(address + slice * 3, 64),
        "a never-uploaded slice counted as CPU clean");

  tracker.MarkRegionAsCpuModified(address + slice + 100, 4);
  Check(tracker.IsRegionCpuCleanHint(address, slice) &&
            tracker.IsRegionCpuCleanHint(address + slice * 2, slice),
        "a CPU write dirtied another slice");
  Check(!tracker.IsRegionCpuCleanHint(address + slice, 64) &&
            !tracker.IsRegionCpuCleanHint(address, slice * 3),
        "a written slice counted as CPU clean");
  upload(address + slice, slice);
  Check(tracker.IsRegionCpuCleanHint(address, slice * 3),
        "uploading the written slice did not make it CPU clean");

  tracker.UntrackMemory(address, slice * 3);
  Check(VirtualFree(memory, 0, MEM_RELEASE) != 0, "VirtualFree failed");
}

void TestRangeInvalidation() {
  constexpr uintptr_t base = 0x0000000201000000ull;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  constexpr uint64_t size = Libs::Graphics::TRACKER_REGION_SIZE * 2;
  auto *memory = static_cast<uint8_t *>(
      VirtualAlloc(reinterpret_cast<void *>(base), size,
                   MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
  Check(memory == reinterpret_cast<void *>(base),
        "range invalidation allocation failed");
  const auto address = reinterpret_cast<uint64_t>(memory);

  tracker.ForEachUploadRange(
      address, size, true, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  Check(tracker.IsRegionGpuModified(address, size) && !IsWritable(memory),
        "range invalidation setup did not establish GPU ownership");
  uint32_t flushes = 0;
  tracker.InvalidateRegion(address + 16, size - 32, [&] {
    flushes++;
    tracker.ForEachDownloadRange<true>(address + 16, size - 32,
                                       [](uint64_t, uint64_t) noexcept {});
    tracker.MarkRegionAsCpuModified(address + 16, size - 32);
  });
  Check(flushes == 1 && !tracker.IsRegionGpuModified(address, size) &&
            tracker.IsRegionCpuModified(address, size) && IsWritable(memory) &&
            IsWritable(memory + size - 1),
        "range invalidation did not batch ownership transfer across regions");
  tracker.InvalidateRegion(address + 16, size - 32, [&] { flushes++; });
  Check(flushes == 1,
        "clean range invalidation unnecessarily requested a GPU flush");
  tracker.UntrackMemory(address, size);
  Release(memory);
}

void TestGpuReacquisitionAfterInvalidation() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 1);
  const auto address = reinterpret_cast<uint64_t>(memory);

  tracker.ForEachUploadRange(
      address, page_size, true, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  Check(tracker.IsRegionGpuModified(address, page_size) &&
            !tracker.IsRegionCpuModified(address, page_size),
        "reacquisition setup did not establish GPU ownership");

  uint32_t flushes = 0;
  uint32_t uploads = 0;
  std::binary_semaphore reacquire{0};
  std::binary_semaphore reacquired{0};
  std::jthread publisher([&] {
    reacquire.acquire();
    tracker.ForEachUploadRange(
        address + 16, 32, true,
        [&](uint64_t, uint64_t) noexcept { uploads++; }, []() noexcept {});
    reacquired.release();
  });
  tracker.InvalidateRegion(address + 16, 32, [&] {
    flushes++;
    tracker.ForEachDownloadRange<true>(address + 16, 32,
                                       [](uint64_t, uint64_t) noexcept {});
    tracker.MarkRegionAsCpuModified(address + 16, 32);
    reacquire.release();
    reacquired.acquire();
  });
  publisher.join();
  Check(flushes == 1 && uploads == 1 &&
            tracker.IsRegionGpuModified(address, page_size) &&
            !tracker.IsRegionCpuModified(address, page_size) &&
            !IsWritable(memory),
        "invalidation rejected a new generation of GPU ownership");

  tracker.UnmarkRegionAsGpuModified(address, page_size);
  tracker.MarkRegionAsCpuModified(address, page_size);
  tracker.UntrackMemory(address, page_size);
  Release(memory);
}

void TestGpuDirtyBits() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 2);
  const auto address = reinterpret_cast<uint64_t>(memory);

  tracker.ForEachUploadRange(
      address, page_size, true, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  Check(tracker.IsRegionGpuModified(address, page_size) &&
            !tracker.IsRegionGpuModified(address + page_size, page_size) &&
            Protection(memory) == PAGE_NOACCESS,
        "GPU dirty state escaped the requested range");
  tracker.UnmarkRegionAsGpuModified(address, page_size);
  Check(!tracker.IsRegionGpuModified(address, page_size) &&
            Protection(memory) == PAGE_READONLY,
        "GPU dirty state did not restore write-only tracking");
  tracker.MarkRegionAsCpuModified(address, page_size);
  tracker.UntrackMemory(address, page_size * 2);
  Release(memory);
}

void TestReadbackArming() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 3);
  const auto address = reinterpret_cast<uint64_t>(memory);
  const auto page = [&](uint64_t index) { return address + index * page_size; };

  tracker.ForEachUploadRange(
      address, page_size * 3, true, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  auto state = tracker.QueryReadback(address, page_size * 3);
  Check(state.gpu_dirty && state.unarmed && state.tick == 0 &&
            !tracker.HasArmedPages(address, page_size * 3),
        "fresh GPU-dirty pages reported an armed readback");

  tracker.ArmReadback(address, page_size * 2, 1, 5);
  state = tracker.QueryReadback(address, page_size * 2);
  Check(state.gpu_dirty && !state.unarmed && state.tick == 5 &&
            tracker.HasArmedPages(page(1), page_size) &&
            !tracker.HasArmedPages(page(2), page_size),
        "arming did not cover exactly the requested pages");
  std::vector<std::pair<uint64_t, uint64_t>> unarmed;
  tracker.ForEachUnarmedDownloadRange(
      address, page_size * 3, [&](uint64_t start, uint64_t bytes) noexcept {
        unarmed.emplace_back(start, bytes);
      });
  Check(unarmed.size() == 1 && unarmed[0].first == page(2) &&
            unarmed[0].second == page_size,
        "armed pages were offered for a second download");

  // A later download arms only what is still unarmed; earlier arms keep their token.
  tracker.ArmReadback(address, page_size * 3, 2, 6);
  state = tracker.QueryReadback(address, page_size);
  Check(!state.unarmed && state.tick == 5 &&
            tracker.QueryReadback(page(2), page_size).tick == 6,
        "re-arming replaced an earlier download's token");

  // A GPU write recorded after the download supersedes it.
  tracker.MarkRegionAsGpuModified(page(1) + 16, 16);
  state = tracker.QueryReadback(page(1), page_size);
  Check(state.gpu_dirty && state.unarmed && !tracker.HasArmedPages(page(1), page_size),
        "a new GPU write did not disarm its page");

  tracker.FinalizeReadback(address, page_size * 3, 1);
  Check(!tracker.IsRegionGpuModified(page(0), page_size) &&
            Protection(memory) == PAGE_READONLY &&
            tracker.IsRegionGpuModified(page(1), page_size) &&
            Protection(memory + page_size) == PAGE_NOACCESS &&
            tracker.IsRegionGpuModified(page(2), page_size) &&
            tracker.HasArmedPages(page(2), page_size),
        "finalizing published a re-dirtied page or another download's page");

  tracker.FinalizeReadback(page(2), page_size, 2);
  Check(!tracker.IsRegionGpuModified(page(2), page_size) &&
            Protection(memory + page_size * 2) == PAGE_READONLY &&
            !tracker.HasArmedPages(address, page_size * 3),
        "finalizing the matching token did not publish its page");

  // An explicit unmark also drops arms, so the armed-page count returns to zero.
  tracker.ArmReadback(page(1), page_size, 3, 7);
  tracker.UnmarkRegionAsGpuModified(page(1), page_size);
  Check(!tracker.HasArmedPages(address, page_size * 3) &&
            !tracker.QueryReadback(address, page_size * 3).gpu_dirty,
        "unmarking left an armed page behind");
  tracker.FinalizeReadback(page(1), page_size, 3);
  Check(!tracker.IsRegionGpuModified(page(1), page_size) &&
            Protection(memory + page_size) == PAGE_READONLY,
        "a stale publication changed an unmarked page");

  tracker.MarkRegionAsCpuModified(address, page_size * 3);
  tracker.UntrackMemory(address, page_size * 3);
  Release(memory);
}

void TestStaleReadGrant() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 2);
  const auto address = reinterpret_cast<uint64_t>(memory);
  const auto second = address + page_size;

  tracker.ForEachUploadRange(
      address, page_size * 2, true, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  // Only armed, hot pages open.
  tracker.SetReadbackHot(address, page_size * 2, true);
  Check(!tracker.GrantStaleRead(address, page_size) &&
            Protection(memory) == PAGE_NOACCESS,
        "an unarmed page was opened for stale reads");
  tracker.ArmReadback(address, page_size * 2, 1, 5);
  tracker.SetReadbackHot(second, page_size, false);
  Check(!tracker.GrantStaleRead(second, page_size) &&
            Protection(memory + page_size) == PAGE_NOACCESS,
        "a page that is not hot was opened for stale reads");
  Check(tracker.GrantStaleRead(address, page_size) &&
            Protection(memory) == PAGE_READONLY &&
            tracker.IsRegionGpuModified(address, page_size) &&
            !tracker.QueryReadback(address, page_size).unarmed,
        "an armed hot page did not open read-only, or lost its GPU state");

  // The next GPU write closes it again and supersedes the download.
  tracker.MarkRegionAsGpuModified(address + 16, 16);
  Check(Protection(memory) == PAGE_NOACCESS &&
            tracker.QueryReadback(address, page_size).unarmed,
        "a GPU write left a stale-readable page open");
  tracker.FinalizeReadback(address, page_size * 2, 1);
  Check(Protection(memory) == PAGE_NOACCESS &&
            tracker.IsRegionGpuModified(address, page_size),
        "an old publication cleared a re-dirtied page");

  // Publication of the current download leaves the page clean and readable.
  tracker.ArmReadback(address, page_size, 2, 6);
  Check(tracker.GrantStaleRead(address, page_size) && Protection(memory) == PAGE_READONLY,
        "a re-armed hot page did not reopen");
  tracker.FinalizeReadback(address, page_size, 2);
  Check(!tracker.IsRegionGpuModified(address, page_size) &&
            Protection(memory) == PAGE_READONLY,
        "publication did not leave the page clean and readable");

  tracker.MarkRegionAsCpuModified(address, page_size * 2);
  tracker.UntrackMemory(address, page_size * 2);
  Release(memory);
}

void TestExactDirtyIntervalsSharingTrackerPage() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 1);
  const auto address = reinterpret_cast<uint64_t>(memory);

  tracker.ForEachUploadRange(
      address, page_size, false, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  RangeSet exact_dirty;
  exact_dirty.Add(address + 64, 16);
  exact_dirty.Add(address + 192, 32);

  ResetProtectionLog();
  tracker.MarkRegionAsGpuModified(address + 64, 16);
  tracker.MarkRegionAsGpuModified(address + 192, 32);
  Check(g_protection_calls == 1 &&
            tracker.IsRegionGpuModified(address, page_size) &&
            Protection(memory) == PAGE_NOACCESS,
        "disjoint byte dirtiness duplicated the page watcher");

  exact_dirty.Subtract(address + 64, 16);
  if (!exact_dirty.Intersects(address, page_size)) {
    tracker.UnmarkRegionAsGpuModified(address, page_size);
  }
  Check(g_protection_calls == 1 &&
            tracker.IsRegionGpuModified(address, page_size) &&
            Protection(memory) == PAGE_NOACCESS,
        "draining one exact interval prematurely released its shared page");

  exact_dirty.Subtract(address + 192, 32);
  if (!exact_dirty.Intersects(address, page_size)) {
    tracker.UnmarkRegionAsGpuModified(address, page_size);
  }
  Check(g_protection_calls == 2 &&
            !tracker.IsRegionGpuModified(address, page_size) &&
            Protection(memory) == PAGE_READONLY,
        "draining the final exact interval did not release its tracker page");

  tracker.UntrackMemory(address, page_size);
  Release(memory);
}

void TestGpuDownloadProtectionMirrors() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 4);
  const auto address = reinterpret_cast<uint64_t>(memory);

  tracker.ForEachUploadRange(
      address, page_size * 4, false, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  tracker.MarkRegionAsGpuModified(address + 16, 32);
  tracker.MarkRegionAsGpuModified(address + page_size * 2 + 16, 32);

  std::vector<std::pair<uint64_t, uint64_t>> visited;
  ResetProtectionLog();
  tracker.ForEachDownloadRange<false>(
      address, page_size * 3,
      [&](uint64_t range_address, uint64_t range_size) noexcept {
        visited.push_back({range_address, range_size});
      });
  Check(visited.size() == 2 && visited[0].first == address &&
            visited[0].second == page_size &&
            visited[1].first == address + page_size * 2 &&
            visited[1].second == page_size && g_protection_calls == 0 &&
            tracker.IsRegionGpuModified(address, page_size * 3),
        "non-clearing download changed protection or lost sparse ranges");

  visited.clear();
  bool protected_during_download = false;
  tracker.ForEachDownloadRange<true>(
      address + 16, 32,
      [&](uint64_t range_address, uint64_t range_size) noexcept {
        protected_during_download = Protection(memory) == PAGE_NOACCESS;
        visited.push_back({range_address, range_size});
      });
  Check(protected_during_download && visited.size() == 1 &&
            visited[0].first == address &&
            visited[0].second == page_size && g_protection_log.size() == 1 &&
            g_protection_log[0].address == address &&
            g_protection_log[0].size == page_size &&
            g_protection_log[0].mode == Common::VirtualMemory::Mode::Read &&
            !tracker.IsRegionGpuModified(address, page_size) &&
            tracker.IsRegionGpuModified(address + page_size * 2, page_size) &&
            Protection(memory) == PAGE_READONLY &&
            Protection(memory + page_size * 2) == PAGE_NOACCESS,
        "partial download did not preserve the CPU/GPU protection mirrors");

  visited.clear();
  ResetProtectionLog();
  tracker.ForEachDownloadRange<true>(
      address + 16, 32,
      [&](uint64_t range_address, uint64_t range_size) noexcept {
        visited.push_back({range_address, range_size});
      });
  Check(visited.empty() && g_protection_calls == 0 &&
            tracker.IsRegionGpuModified(address + page_size * 2, page_size),
        "idempotent partial download disturbed another GPU-owned page");

  tracker.UnmarkRegionAsGpuModified(address, page_size * 3);
  Check(!tracker.IsRegionGpuModified(address, page_size * 3) &&
            Protection(memory + page_size * 2) == PAGE_READONLY,
        "broad final unmark did not restore write-only tracking");
  ResetProtectionLog();
  tracker.MarkRegionAsCpuModified(address + 16, 32);
  Check(
      g_protection_log.size() == 1 && g_protection_log[0].address == address &&
          g_protection_log[0].size == page_size &&
          g_protection_log[0].mode == Common::VirtualMemory::Mode::ReadWrite &&
          IsWritable(memory) && !IsWritable(memory + page_size),
      "CPU-dirty transition did not release only its write watcher");

  tracker.UntrackMemory(address, page_size * 4);
  Release(memory);
}

void TestCrossRegionUpload() {
  constexpr uintptr_t base = 0x0000000200010000ull;
  constexpr uint64_t region_size = 4ull * 1024ull * 1024ull;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = static_cast<uint8_t *>(
      VirtualAlloc(reinterpret_cast<void *>(base), region_size * 2,
                   MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
  Check(memory == reinterpret_cast<void *>(base), "fixed VirtualAlloc failed");
  const auto address = reinterpret_cast<uint64_t>(memory);
  const auto boundary = (address + region_size - 1) & ~(region_size - 1);
  uint32_t ranges = 0;
  tracker.ForEachUploadRange(
      boundary - page_size, page_size * 2, false,
      [&](uint64_t, uint64_t) noexcept { ranges++; }, []() noexcept {});
  Check(ranges == 2 &&
            !tracker.IsRegionCpuModified(boundary - page_size, page_size * 2) &&
            !IsWritable(reinterpret_cast<void *>(boundary - page_size)) &&
            !IsWritable(reinterpret_cast<void *>(boundary)),
        "cross-region upload did not clear and protect both regions");
  tracker.MarkRegionAsCpuModified(boundary - page_size, page_size * 2);
  tracker.UntrackMemory(address, region_size * 2);
  Release(memory);
}

void TestUploadDoesNotSerializeDisjointRegion() {
  constexpr auto region_size = Libs::Graphics::TRACKER_REGION_SIZE;
  constexpr auto page_size = Libs::Graphics::TRACKER_PAGE_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  auto *memory = Allocate(page_manager, region_size * 2 / page_size);
  const auto allocation_base = reinterpret_cast<uint64_t>(memory);
  const auto second_region =
      (allocation_base & ~(region_size - 1)) + region_size;
  Check(second_region + page_size <= allocation_base + region_size * 2,
        "test allocation does not span two tracker regions");

  // Publish both managers before the concurrent section so this test measures
  // tracker access serialization rather than manager allocation.
  Check(tracker.IsRegionCpuModified(allocation_base, page_size) &&
            tracker.IsRegionCpuModified(second_region, page_size),
        "disjoint upload setup did not initialize both regions");

  std::binary_semaphore upload_entered{0};
  std::binary_semaphore finish_upload{0};
  std::binary_semaphore query_finished{0};
  std::atomic_bool query_result{false};
  std::jthread uploader([&] {
    tracker.ForEachUploadRange(
        allocation_base, page_size, true, [](uint64_t, uint64_t) noexcept {},
        [&]() noexcept {
          upload_entered.release();
          finish_upload.acquire();
        });
  });
  upload_entered.acquire();
  std::jthread query([&] {
    query_result.store(tracker.IsRegionCpuModified(second_region, page_size),
                       std::memory_order_relaxed);
    query_finished.release();
  });

  const bool completed_while_upload_blocked =
      query_finished.try_acquire_for(std::chrono::seconds(5));
  finish_upload.release();
  uploader.join();
  query.join();

  tracker.UnmarkRegionAsGpuModified(allocation_base, page_size);
  tracker.MarkRegionAsCpuModified(allocation_base, page_size);
  tracker.UntrackMemory(allocation_base, region_size * 2);
  Release(memory);
  Check(completed_while_upload_blocked &&
            query_result.load(std::memory_order_relaxed),
        "upload callback serialized an unrelated tracker region");
}

void TestDownloadDoesNotSerializeDisjointRegion() {
  constexpr auto region_size = Libs::Graphics::TRACKER_REGION_SIZE;
  constexpr auto page_size = Libs::Graphics::TRACKER_PAGE_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  auto *memory = Allocate(page_manager, region_size * 2 / page_size);
  const auto allocation_base = reinterpret_cast<uint64_t>(memory);
  const auto second_region =
      (allocation_base & ~(region_size - 1)) + region_size;
  Check(second_region + page_size <= allocation_base + region_size * 2,
        "test allocation does not span two tracker regions");

  tracker.ForEachUploadRange(
      allocation_base, page_size, true, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  tracker.ForEachUploadRange(
      second_region, page_size, false, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});

  std::binary_semaphore download_entered{0};
  std::binary_semaphore finish_download{0};
  std::binary_semaphore mutation_finished{0};
  std::jthread downloader([&] {
    tracker.ForEachDownloadRange<false>(
        allocation_base, second_region + page_size - allocation_base,
        [&](uint64_t address, uint64_t) noexcept {
          if (address == allocation_base) {
            download_entered.release();
            finish_download.acquire();
          }
        });
  });
  download_entered.acquire();
  std::jthread mutation([&] {
    tracker.MarkRegionAsGpuModified(second_region, page_size);
    mutation_finished.release();
  });

  const bool completed_while_download_blocked =
      mutation_finished.try_acquire_for(std::chrono::seconds(5));
  finish_download.release();
  downloader.join();
  mutation.join();

  const bool both_gpu_owned =
      tracker.IsRegionGpuModified(allocation_base, page_size) &&
      tracker.IsRegionGpuModified(second_region, page_size);
  tracker.UnmarkRegionAsGpuModified(allocation_base, page_size);
  tracker.UnmarkRegionAsGpuModified(second_region, page_size);
  tracker.MarkRegionAsCpuModified(allocation_base, page_size);
  tracker.MarkRegionAsCpuModified(second_region, page_size);
  tracker.UntrackMemory(allocation_base, region_size * 2);
  Release(memory);
  Check(completed_while_download_blocked && both_gpu_owned,
        "download callback serialized an unrelated tracker region");
}

void TestGpuUnmarkUsesRegionMask() {
  constexpr auto region_size = Libs::Graphics::TRACKER_REGION_SIZE;
  constexpr auto page_size = Libs::Graphics::TRACKER_PAGE_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  auto *memory = Allocate(page_manager, region_size * 2 / page_size);
  const auto allocation_base = reinterpret_cast<uint64_t>(memory);
  const auto region_base =
      (allocation_base + region_size - 1) & ~(region_size - 1);
  Check(region_base + region_size + page_size <=
            allocation_base + region_size * 2,
        "test allocation does not span two complete tracker regions");

  const auto sparse_begin = region_base + page_size;
  tracker.ForEachUploadRange(
      sparse_begin, page_size * 3, false, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  tracker.MarkRegionAsGpuModified(sparse_begin, page_size);
  tracker.MarkRegionAsGpuModified(sparse_begin + page_size * 2, page_size);
  ResetProtectionLog();
  tracker.UnmarkRegionAsGpuModified(sparse_begin, page_size * 3);
  Check(
      g_protection_calls == 1 && g_protection_log.size() == 1 &&
          g_protection_log[0].address == sparse_begin &&
          g_protection_log[0].size == page_size * 3 &&
          g_protection_log[0].mode == Common::VirtualMemory::Mode::Read &&
          !tracker.IsRegionGpuModified(sparse_begin, page_size * 3) &&
          Protection(reinterpret_cast<void *>(sparse_begin)) == PAGE_READONLY &&
          Protection(reinterpret_cast<void *>(sparse_begin + page_size)) ==
              PAGE_READONLY &&
          Protection(reinterpret_cast<void *>(sparse_begin + page_size * 2)) ==
              PAGE_READONLY,
      "GPU unmark did not coalesce a sparse 4 MiB region mask");
  ResetProtectionLog();
  tracker.UnmarkRegionAsGpuModified(sparse_begin, page_size * 3);
  Check(g_protection_calls == 0,
        "idempotent GPU unmark performed a protection call");

  const auto boundary = region_base + region_size;
  const auto cross_begin = boundary - page_size;
  tracker.ForEachUploadRange(
      cross_begin, page_size * 2, false, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  tracker.MarkRegionAsGpuModified(cross_begin, page_size * 2);
  ResetProtectionLog();
  tracker.UnmarkRegionAsGpuModified(cross_begin, page_size * 2);
  Check(g_protection_calls == 2 && g_protection_log.size() == 2 &&
            g_protection_log[0].address == cross_begin &&
            g_protection_log[0].size == page_size &&
            g_protection_log[0].mode == Common::VirtualMemory::Mode::Read &&
            g_protection_log[1].address == boundary &&
            g_protection_log[1].size == page_size &&
            g_protection_log[1].mode == Common::VirtualMemory::Mode::Read &&
            !tracker.IsRegionGpuModified(cross_begin, page_size * 2),
        "cross-region GPU unmark did not use one update per 4 MiB region");

  tracker.UntrackMemory(allocation_base, region_size * 2);
  Release(memory);
}

void TestFullRegionGpuUnmarkBatching() {
  constexpr auto region_size = Libs::Graphics::TRACKER_REGION_SIZE;
  constexpr auto page_size = Libs::Graphics::TRACKER_PAGE_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  auto *memory = Allocate(page_manager, region_size * 2 / page_size);
  const auto allocation_base = reinterpret_cast<uint64_t>(memory);
  const auto region_base =
      (allocation_base + region_size - 1) & ~(region_size - 1);
  Check(region_base + region_size <= allocation_base + region_size * 2,
        "test allocation does not contain a complete tracker region");

  tracker.ForEachUploadRange(
      region_base, region_size, false, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  tracker.MarkRegionAsGpuModified(region_base, region_size);
  Check(tracker.IsRegionGpuModified(region_base, region_size) &&
            Protection(reinterpret_cast<void *>(region_base)) ==
                PAGE_NOACCESS &&
            Protection(reinterpret_cast<void *>(region_base + region_size -
                                                page_size)) == PAGE_NOACCESS,
        "full-region setup did not establish GPU read protection");

  ResetProtectionLog();
  tracker.UnmarkRegionAsGpuModified(region_base, region_size);
  Check(
      g_protection_log.size() == 1 &&
          g_protection_log[0].address == region_base &&
          g_protection_log[0].size == region_size &&
          g_protection_log[0].mode == Common::VirtualMemory::Mode::Read &&
          !tracker.IsRegionGpuModified(region_base, region_size) &&
          Protection(reinterpret_cast<void *>(region_base)) == PAGE_READONLY &&
          Protection(reinterpret_cast<void *>(region_base + region_size -
                                              page_size)) == PAGE_READONLY,
      "full-region GPU unmark did not use one exact 4 MiB protection request");

  tracker.UntrackMemory(allocation_base, region_size * 2);
  Release(memory);
}

void TestBdaHintPublication() {
  constexpr auto region_size = Libs::Graphics::TRACKER_REGION_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  tracker.PublishBdaHints(region_size * 64 - 1, 2);
  Check(tracker.ConsumeBdaHintWord(0) == (uint64_t{1} << 63) &&
            tracker.ConsumeBdaHintWord(1) == 1,
        "cross-word publication lost one region");
  tracker.PublishBdaHints(TRACKER_ADDRESS_SIZE - 1, UINT64_MAX);
  Check(tracker.ConsumeBdaHintWord(MemoryTracker::BDA_HINT_WORDS - 1) ==
            (uint64_t{1} << 63),
        "last-address publication overflowed the hint index");

  constexpr uint64_t address = 0x0000000203000000ull;
  const auto region = address / region_size;
  Check(tracker.FindRegion(region) == nullptr,
        "hint publication unnecessarily allocated a tracker region");
  tracker.PublishBdaHints(address, 1);
  const auto claimed = tracker.ConsumeBdaHintWord(region / 64);
  Check(claimed == (uint64_t{1} << (region % 64)),
        "untracked registration did not publish its region");
  Check(tracker.IsRegionCpuModified(address, 1) &&
            tracker.FindRegion(region) != nullptr &&
            tracker.IsBdaHintPending(region),
        "new manager became visible without a CPU-dirty hint");

  // A failed pass must merge its unfinished claim with a different concurrent
  // publication.
  (void)tracker.ConsumeBdaHintWord(region / 64);
  tracker.PublishBdaHints(address + region_size, 1);
  tracker.RestoreBdaHints(region / 64, claimed);
  Check(tracker.IsBdaHintPending(region) &&
            tracker.IsBdaHintPending(region + 1),
        "restoring an unfinished claim overwrote a concurrent hint");
}

void TestBdaHintRaces() {
  constexpr auto region_size = Libs::Graphics::TRACKER_REGION_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  const auto page_size = harness.page_manager.GetPageSize();
  auto *memory = Allocate(harness.page_manager, 2);
  const auto address = reinterpret_cast<uint64_t>(memory);
  const auto region = address / region_size;
  const auto page = (address % region_size) / Libs::Graphics::TRACKER_PAGE_SIZE;
  const auto mask = uint64_t{1} << (region % 64);
  memory[0] = 7;
  tracker.ForEachUploadRange(
      address, page_size * 2, false, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  (void)tracker.ConsumeBdaHintWord(region / 64);
  auto *manager = tracker.FindRegion(region);
  Check(manager != nullptr, "upload failed to create its tracker region");

  // The consumer has already snapshotted a clean page when a later CPU write
  // arrives.
  const auto clean_snapshot = tracker.SnapshotCpuDirty(*manager);
  Check(!clean_snapshot.Get(page),
        "initial BDA snapshot was unexpectedly dirty");
  std::binary_semaphore publish{0};
  std::binary_semaphore published{0};
  std::jthread writer([&] {
    publish.acquire();
    tracker.MarkRegionAsCpuModified(address, page_size);
    memory[0] = 11;
    published.release();
  });
  publish.release();
  published.acquire();
  writer.join();
  Check(tracker.IsBdaHintPending(region) &&
            tracker.BdaHintsCoverCpuDirty(address, page_size),
        "write after a clean snapshot lost its next-pass hint");

  Check((tracker.ConsumeBdaHintWord(region / 64) & mask) != 0,
        "next BDA pass did not claim the racing write");
  Check(tracker.SnapshotCpuDirty(*manager).Get(page),
        "next BDA snapshot did not include the racing write");

  // A second write after the upload copies must survive the current pass
  // completing.
  uint8_t uploaded_value = 0;
  std::binary_semaphore copied{0};
  std::binary_semaphore rewritten{0};
  std::jthread rewriter([&] {
    copied.acquire();
    tracker.MarkRegionAsCpuModified(address, page_size);
    memory[0] = 23;
    rewritten.release();
  });
  tracker.ForEachUploadRange(
      address, page_size, false, [](uint64_t, uint64_t) noexcept {},
      [&]() noexcept {
        uploaded_value = memory[0];
        copied.release();
        rewritten.acquire();
      });
  rewriter.join();
  Check(uploaded_value == 11 &&
            tracker.IsRegionCpuModified(address, page_size) &&
            tracker.IsBdaHintPending(region),
        "write after upload copy was cleared by the completed pass");

  // Republishing an already-dirty page after exchange must also remain visible.
  (void)tracker.ConsumeBdaHintWord(region / 64);
  tracker.MarkRegionAsCpuModified(address, page_size);
  Check(tracker.IsBdaHintPending(region),
        "already-dirty write did not republish its consumed hint");
  (void)tracker.ConsumeBdaHintWord(region / 64);
  tracker.ForEachUploadRange(
      address, page_size, false, [](uint64_t, uint64_t) noexcept {},
      [&]() noexcept { uploaded_value = memory[0]; });
  Check(uploaded_value == 23 &&
            !tracker.IsRegionCpuModified(address, page_size) &&
            tracker.BdaHintsCoverCpuDirty(address, page_size),
        "subsequent upload did not observe the latest CPU bytes");

  tracker.InvalidateRegion(address, page_size,
                           [] { Check(false, "unexpected GPU producer"); });
  Check(tracker.IsBdaHintPending(region),
        "fault invalidation did not publish a CPU-dirty hint");
  (void)tracker.ConsumeBdaHintWord(region / 64);
  tracker.UntrackMemory(address, page_size * 2);
  Check(tracker.IsBdaHintPending(region),
        "untracking did not republish dirty pages");
  Release(memory);
}

// Mirrors BufferCache::SynchronizeBdaSelective's claim order: summary word, then hint words.
std::vector<uint64_t> ClaimSelectivePass(MemoryTracker &tracker) {
  std::vector<uint64_t> claimed(MemoryTracker::BDA_HINT_WORDS);
  for (size_t summary = 0; summary < MemoryTracker::BDA_SUMMARY_WORDS;
       ++summary) {
    for (auto words = tracker.ConsumeBdaSummaryWord(summary); words != 0;
         words &= words - 1) {
      const auto word = summary * 64 + std::countr_zero(words);
      claimed[word] |= tracker.ConsumeBdaHintWord(word);
    }
  }
  return claimed;
}

void TestBdaHintSummary() {
  constexpr auto region_size = Libs::Graphics::TRACKER_REGION_SIZE;
  constexpr auto word_span = region_size * 64;
  TrackerHarness harness;
  auto &tracker = harness.tracker;

  // A range crossing both a hint-word and a summary-word boundary.
  const uint64_t boundary = word_span * 64;
  tracker.PublishBdaHints(boundary - 1, 2);
  Check(tracker.ConsumeBdaSummaryWord(0) == (uint64_t{1} << 63) &&
            tracker.ConsumeBdaSummaryWord(1) == 1,
        "cross-summary publication lost one summary bit");
  Check(tracker.ConsumeBdaHintWord(63) == (uint64_t{1} << 63) &&
            tracker.ConsumeBdaHintWord(64) == 1,
        "cross-summary publication lost one hint bit");
  tracker.PublishBdaHints(TRACKER_ADDRESS_SIZE - 1, 1);
  Check(tracker.ConsumeBdaSummaryWord(MemoryTracker::BDA_SUMMARY_WORDS - 1) ==
            (uint64_t{1} << 63),
        "last-address publication overflowed the summary index");
  (void)tracker.ConsumeBdaHintWord(MemoryTracker::BDA_HINT_WORDS - 1);
  Check(ClaimSelectivePass(tracker) ==
            std::vector<uint64_t>(MemoryTracker::BDA_HINT_WORDS),
        "consumed hints remained visible to a selective pass");

  constexpr uint64_t address = 0x0000000203000000ull;
  const auto region = address / region_size;
  const auto word = region / 64;
  const auto bit = uint64_t{1} << (region % 64);
  tracker.PublishBdaHints(address, 1);
  // A pass that claimed the summary but not yet the hint word: a republished
  // hint must restore the summary bit so a later pass finds it.
  (void)tracker.ConsumeBdaSummaryWord(word / 64);
  Check(!tracker.IsBdaHintPending(region),
        "claimed summary still reported the region pending");
  tracker.PublishBdaHints(address, 1);
  Check(tracker.IsBdaHintPending(region),
        "republished hint did not restore its claimed summary bit");
  auto claimed = ClaimSelectivePass(tracker);
  Check(claimed[word] == bit, "selective pass did not claim the hint");

  // Restoring an unfinished claim also re-publishes its summary bit.
  tracker.RestoreBdaHints(word, bit);
  Check(tracker.IsBdaHintPending(region),
        "restored claim was not reachable from its summary");
  (void)ClaimSelectivePass(tracker);
  tracker.RestoreBdaSummary(word / 64, uint64_t{1} << (word % 64));
  Check(ClaimSelectivePass(tracker)[word] == 0,
        "restored summary bit reported a hint that was already claimed");

  // No publication may be lost while passes run concurrently.
  constexpr int publishers = 4;
  constexpr int publications = 20000;
  std::atomic<int> finished{0};
  std::vector<std::vector<uint64_t>> published(
      publishers, std::vector<uint64_t>(MemoryTracker::BDA_HINT_WORDS));
  std::vector<uint64_t> total(MemoryTracker::BDA_HINT_WORDS);
  {
    std::vector<std::jthread> threads;
    for (int thread = 0; thread < publishers; ++thread) {
      threads.emplace_back([&, thread] {
        uint64_t state = 0x9e3779b97f4a7c15ull * (thread + 1);
        for (int index = 0; index < publications; ++index) {
          state = state * 6364136223846793005ull + 1442695040888963407ull;
          // Concentrate on a few summary words so publishers and the consumer
          // contend on the same summary and hint words.
          const auto target = (state >> 20) % (64 * 64 * 3);
          tracker.PublishBdaHints(target * region_size, 1);
          published[thread][target / 64] |= uint64_t{1} << (target % 64);
        }
        finished.fetch_add(1);
      });
    }
    while (finished.load() != publishers) {
      const auto pass = ClaimSelectivePass(tracker);
      for (size_t index = 0; index < total.size(); ++index) {
        total[index] |= pass[index];
      }
    }
  }
  const auto last = ClaimSelectivePass(tracker);
  for (size_t index = 0; index < total.size(); ++index) {
    total[index] |= last[index];
    uint64_t expected = 0;
    for (const auto &thread : published) {
      expected |= thread[index];
    }
    Check((total[index] & expected) == expected,
          "concurrent selective passes lost a published hint");
  }
}

[[noreturn]] void RunDeathCase(const char *name) {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 1);
  const auto address = reinterpret_cast<uint64_t>(memory);
  if (std::strcmp(name, "gpu-dirty-explicit-cpu") == 0) {
    tracker.ForEachUploadRange(
        address, page_size, true, [](uint64_t, uint64_t) noexcept {},
        []() noexcept {});
    tracker.MarkRegionAsCpuModified(address, page_size);
  } else if (std::strcmp(name, "reentrant-upload") == 0) {
    tracker.ForEachUploadRange(
        address, page_size, true, [](uint64_t, uint64_t) noexcept {},
        [&]() noexcept {
          (void)tracker.IsRegionCpuModified(address, page_size);
        });
  }
  std::_Exit(0x7f);
}

void CheckDeathCase(const char *name) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
  char path[MAX_PATH]{};
  Check(GetModuleFileNameA(nullptr, path, MAX_PATH) != 0,
        "GetModuleFileName failed");
  std::string command = std::string("\"") + path + "\" --death " + name;
  std::vector<char> mutable_command(command.begin(), command.end());
  mutable_command.push_back('\0');
  STARTUPINFOA startup{sizeof(startup)};
  PROCESS_INFORMATION process{};
  Check(CreateProcessA(nullptr, mutable_command.data(), nullptr, nullptr, FALSE,
                       CREATE_NO_WINDOW, nullptr, nullptr, &startup,
                       &process) != 0,
        "CreateProcess failed");
  Check(WaitForSingleObject(process.hProcess, 10000) == WAIT_OBJECT_0,
        "MemoryTracker death test timed out");
  DWORD exit_code = 0;
  Check(
      GetExitCodeProcess(process.hProcess, &exit_code) != 0 &&
          (exit_code == 321 || exit_code == EXCEPTION_NONCONTINUABLE_EXCEPTION),
      "MemoryTracker death path used the wrong exit");
  CloseHandle(process.hThread);
  CloseHandle(process.hProcess);
#else
  const pid_t pid = ::fork();
  Check(pid >= 0, "fork failed");
  if (pid == 0) {
    ::execl("/proc/self/exe", "MemoryTrackerTests", "--death", name, nullptr);
    std::_Exit(0x7e);
  }
  int status = 0;
  Check(::waitpid(pid, &status, 0) == pid, "waitpid failed");
  const bool fatal_exit =
      WIFEXITED(status) && WEXITSTATUS(status) == (321 & 0xff);
  Check(fatal_exit || WIFSIGNALED(status),
        "MemoryTracker death path used the wrong exit");
#endif
}

void TestFatalPaths() {
  for (const char *name : {"gpu-dirty-explicit-cpu", "reentrant-upload"}) {
    CheckDeathCase(name);
  }
}

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX
void *g_fault_stack = nullptr;
constexpr size_t FAULT_STACK_SIZE = 64 * 1024;
volatile sig_atomic_t g_stack_faults = 0;

bool HandleStackFault(const Common::HostException::ExceptionInfo &info) {
  using namespace Common::HostException;
  stack_t active_stack{};
  const auto fault_address = reinterpret_cast<uintptr_t>(g_fault_stack) +
                             FAULT_STACK_SIZE - sizeof(uintptr_t);
  if (info.type != ExceptionType::AccessViolation ||
      info.access_violation_type != AccessViolationType::Write ||
      info.access_violation_vaddr != fault_address ||
      ::sigaltstack(nullptr, &active_stack) != 0 ||
      (active_stack.ss_flags & SS_ONSTACK) == 0) {
    std::_Exit(1);
  }
  g_stack_faults = 1;
  return ::mprotect(g_fault_stack, FAULT_STACK_SIZE,
                    PROT_READ | PROT_WRITE) == 0;
}

// A stack write must fault before any signal frame can use the protected stack.
__attribute__((naked)) void WriteProtectedStack(void *) {
  asm volatile("mov %rsp, %rax\n"
               "mov %rdi, %rsp\n"
               "push %rax\n"
               "pop %rsp\n"
               "ret\n");
}

void TestFaultOnProtectedStack() {
  const pid_t pid = ::fork();
  Check(pid >= 0, "stack fault fork failed");
  if (pid == 0) {
    std::thread worker([] {
      Check(Common::HostException::InitializeThreadSignalStack(),
            "initialize thread signal stack failed");
      Check(Common::HostException::InstallHandler(HandleStackFault),
            "install stack fault handler failed");
      g_fault_stack = ::mmap(nullptr, FAULT_STACK_SIZE, PROT_READ,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      Check(g_fault_stack != MAP_FAILED, "allocate protected stack failed");
      WriteProtectedStack(static_cast<char *>(g_fault_stack) + FAULT_STACK_SIZE);
      Check(g_stack_faults == 1, "protected stack write did not resume");
      struct sigaction action{};
      Check(::sigaction(SIGSEGV, nullptr, &action) == 0 &&
                action.sa_handler != SIG_DFL,
            "stack fault reset the process handler");
      Check(::munmap(g_fault_stack, FAULT_STACK_SIZE) == 0,
            "release protected stack failed");
    });
    worker.join();
    std::_Exit(0);
  }
  int status = 0;
  Check(::waitpid(pid, &status, 0) == pid && WIFEXITED(status) &&
            WEXITSTATUS(status) == 0,
        "fault on a protected stack did not recover");
}
#endif

} // namespace

namespace Libs::LibKernel::Memory {

bool ProtectGuestHostMemory(uint64_t vaddr, uint64_t size,
                            Common::VirtualMemory::Mode mode) {
  return ProtectAddressSpace(vaddr, size, mode);
}

} // namespace Libs::LibKernel::Memory

int main(int argc, char **argv) {
  if (argc == 3 && std::strcmp(argv[1], "--death") == 0) {
    RunDeathCase(argv[2]);
  }
  TestGuestRange();
  TestRangeSet();
  TestQueriesDoNotRequireMappedOwnership();
  TestConcurrentRegionPublication();
  TestCpuDirtyUpload();
  TestCpuCleanHint();
  TestRangeInvalidation();
  TestGpuReacquisitionAfterInvalidation();
  TestGpuDirtyBits();
  TestReadbackArming();
  TestStaleReadGrant();
  TestExactDirtyIntervalsSharingTrackerPage();
  TestGpuDownloadProtectionMirrors();
  TestCrossRegionUpload();
  TestUploadDoesNotSerializeDisjointRegion();
  TestDownloadDoesNotSerializeDisjointRegion();
  TestGpuUnmarkUsesRegionMask();
  TestFullRegionGpuUnmarkBatching();
  TestBdaHintPublication();
  TestBdaHintRaces();
  TestBdaHintSummary();
  TestFatalPaths();
#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX
  TestFaultOnProtectedStack();
#endif
  std::puts("MemoryTrackerTests: all cases passed");
  return 0;
}
