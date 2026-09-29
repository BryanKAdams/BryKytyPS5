#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#include "common/emulatorConfig.h"
#include "common/archive.h"
#include "common/file.h"
#include "ArchiveTestFixture.h"
#include "common/logging/log.h"
#include "common/subsystems.h"
#include "common/threads.h"
#include "common/stringUtils.h"
#include "graphics/presentation/window/windowInternal.h"
#include "kernel/fileSystem.h"
#include "libs/errno.h"
#include "libs/network.h"
#include "loader/symbolDatabase.h"

#include <algorithm>
#include <array>

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <future>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace Libs::LibKernelApr {
void InitLibKernel_1_Apr(Loader::SymbolDatabase *symbols);
}

namespace Libs::LibNet {
void InitNet_1_Net(Loader::SymbolDatabase *symbols);
}

namespace {

namespace FileSystem = Libs::LibKernel::FileSystem;

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "KernelFileSystemTests: failed: %s\n", text);
    std::abort();
  }
}

class TempDirectory {
public:
  TempDirectory() {
    const auto unique =
        std::chrono::steady_clock::now().time_since_epoch().count();
    m_path = std::filesystem::temp_directory_path() /
             ("kyty_kernel_file_system_" + std::to_string(unique));
    Check(std::filesystem::create_directories(m_path),
          "create temporary directory");
  }

  ~TempDirectory() {
    std::error_code error;
    std::filesystem::remove_all(m_path, error);
  }

  [[nodiscard]] const std::filesystem::path &Path() const { return m_path; }

  KYTY_CLASS_NO_COPY(TempDirectory);

private:
  std::filesystem::path m_path;
};

void CheckSaveRename(const std::filesystem::path &root,
                     std::string_view payload) {
  constexpr char Source[] = "/savedata0/STEMP000.DAT";
  constexpr char Target[] = "/savedata0/SDATA000.DAT";
  constexpr char Suffix[] = "-after-rename";

  const int fd = FileSystem::KernelOpen(Source, 0x601, 0777);
  Check(fd >= 3, "open temporary save file");
  Check(FileSystem::KernelWrite(fd, payload.data(), payload.size()) ==
            payload.size(),
        "write save payload");
  Check(FileSystem::KernelRename(Source, Target) == OK,
        "rename open save file");
  Check(FileSystem::KernelWrite(fd, Suffix, sizeof(Suffix) - 1) ==
            sizeof(Suffix) - 1,
        "write through renamed descriptor");
  Check(FileSystem::KernelClose(fd) == OK, "close renamed descriptor");

  Common::File result(root / "SDATA000.DAT", Common::File::Mode::Read);
  Check(!result.IsInvalid(), "open renamed save file");
  const auto data = result.ReadWholeBuffer();
  const std::string expected = std::string(payload) + Suffix;
  Check(data.size() == expected.size(), "renamed save size");
  Check(std::memcmp(data.data(), expected.data(), expected.size()) == 0,
        "renamed save contents");
}

void TestSaveOpenVisibility() {
  constexpr char Path[] = "/savedata0/visible-save.dat";
  constexpr char Payload[] = "saved progress";

  const int fd = FileSystem::KernelOpen(Path, 0xa01, 0777);
  Check(fd >= 3, "create save file exclusively");
  FileSystem::FileStat stat {};
  Check(FileSystem::KernelStat(Path, &stat) == OK && stat.st_size == 0,
        "created save file is visible before close");
  Check(FileSystem::KernelOpen(Path, 0xa01, 0777) ==
            Libs::LibKernel::KERNEL_ERROR_EEXIST,
        "exclusive creation detects an open save file");
  Check(FileSystem::KernelWrite(fd, Payload, sizeof(Payload) - 1) ==
            sizeof(Payload) - 1,
        "populate save file before truncation");
  Check(FileSystem::KernelClose(fd) == OK, "close populated save file");
  Check(FileSystem::KernelStat(Path, &stat) == OK &&
            stat.st_size == sizeof(Payload) - 1,
        "save file contains the truncation fixture");

  const int truncated = FileSystem::KernelOpen(Path, 0x401, 0777);
  Check(truncated >= 3, "truncate existing save file");
  Check(FileSystem::KernelStat(Path, &stat) == OK && stat.st_size == 0,
        "save truncation is visible before close");
  Check(FileSystem::KernelClose(truncated) == OK, "close truncated save file");
}

void CheckMountRoot(const std::filesystem::path &root) {
  Common::File cache;
  Check(cache.Create(root / "rpf.cache"), "create directory listing fixture");
  cache.Close();
  FileSystem::Mount(root, "/app0");
  Check(FileSystem::GetRealFilename("/app0/rpf.cache") == root / "rpf.cache",
        "resolve mount descendant");
  Check(FileSystem::GetRealFilename("/app01/rpf.cache").empty(),
        "mount prefix must end at a path component");

  for (const char *path : {"/app0", "/app0/"}) {
    for (const int flags : {0, 0x00020000}) {
      const int fd = FileSystem::KernelOpen(path, flags, 0);
      Check(fd >= 3, "open mounted root with O_RDONLY or O_DIRECTORY");
      std::array<char, 512> entries {};
      const int size = FileSystem::KernelGetdents(fd, entries.data(), entries.size());
      Check(size > 0 && size <= entries.size(), "enumerate mounted root");
      bool found = false;
      for (int offset = 0; offset < size;) {
        // Directory record: inode, record length, type, name length, name.
        Check(size - offset >= 8, "directory record header fits");
        uint16_t length = 0;
        std::memcpy(&length, entries.data() + offset + 4, sizeof(length));
        const auto name_length = static_cast<uint8_t>(entries[offset + 7]);
        Check(length >= 8 + name_length + 1 && length <= size - offset,
              "directory record and name fit");
        if (std::string_view(entries.data() + offset + 8, name_length) == "rpf.cache") {
          Check(entries[offset + 6] == 8, "cache directory entry is a regular file");
          found = true;
        }
        offset += length;
      }
      Check(found, "mounted root listing contains rpf.cache");
      Check(FileSystem::KernelClose(fd) == OK, "close mounted root");
    }
  }
  FileSystem::Umount("/app0");
  Check(FileSystem::GetRealFilename("/app0/rpf.cache").empty(),
        "unmount by guest path");
  for (const auto &folder : {root, root / ""}) {
    for (const auto &host : {root, root / ""}) {
      FileSystem::Mount(folder, "/app0");
      FileSystem::Umount(Common::PathToGenericString(host));
      Check(FileSystem::GetRealFilename("/app0/rpf.cache").empty(),
            "unmount by host path with or without trailing separator");
    }
  }
}

void CheckUnmappedPaths(const std::filesystem::path &root) {
  const auto host_file = root / "host-only.dat";
  const auto host_path = Common::PathToGenericString(host_file);
  Common::File fixture;
  Check(fixture.Create(host_file), "create unmapped host file");
  fixture.Close();

  FileSystem::FileStat stat {};
  Check(FileSystem::GetRealFilename(host_path).empty() &&
            FileSystem::KernelOpen(host_path.c_str(), 0, 0) ==
                Libs::LibKernel::KERNEL_ERROR_ENOENT &&
            FileSystem::KernelStat(host_path.c_str(), &stat) ==
                Libs::LibKernel::KERNEL_ERROR_ENOENT &&
            FileSystem::KernelCheckReachability(host_path.c_str()) ==
                Libs::LibKernel::KERNEL_ERROR_ENOENT,
        "existing host files are absent from the guest namespace");
  Check(FileSystem::KernelUnlink(host_path.c_str()) ==
            Libs::LibKernel::KERNEL_ERROR_ENOENT &&
            FileSystem::KernelRmdir(Common::PathToGenericString(root).c_str()) ==
                Libs::LibKernel::KERNEL_ERROR_ENOENT &&
            std::filesystem::exists(host_file),
        "unmapped host files and directories cannot be removed");

  const auto missing = Common::PathToGenericString(root / "unmapped-create");
  Check(FileSystem::KernelOpen(missing.c_str(), 0x601, 0777) ==
            Libs::LibKernel::KERNEL_ERROR_ENOENT &&
            FileSystem::KernelMkdir(missing.c_str(), 0777) ==
                Libs::LibKernel::KERNEL_ERROR_ENOENT &&
            !std::filesystem::exists(root / "unmapped-create"),
        "creation requires a mounted guest destination");

  FileSystem::Mount(root, "/app0");
  Check(FileSystem::KernelRename("/app0/host-only.dat", missing.c_str()) ==
            Libs::LibKernel::KERNEL_ERROR_ENOENT &&
            std::filesystem::exists(host_file) &&
            !std::filesystem::exists(root / "unmapped-create"),
        "rename to an unmapped destination preserves the source");
  FileSystem::Umount("/app0");
}

void CheckArchiveMount(const std::filesystem::path &root) {
  const auto archive = root / u8"game-日本語.zar";
  std::vector<uint8_t> payload(2 * 64 * 1024 + 33);
  for (size_t i = 0; i < payload.size(); ++i) {
    payload[i] = static_cast<uint8_t>(i * 37 + 11);
  }
  Check(ArchiveTests::CreateArchive(archive, payload), "create mounted archive fixture");
  const auto archive_root = Common::MakeArchivePath(archive);
  const auto host_member = archive_root / "assets/subdir/data.bin";
  constexpr char GuestMember[] = "/app0/assets/subdir/data.bin";
  FileSystem::Mount(archive_root, "/app0");
  Check(FileSystem::GetRealFilename(GuestMember) == host_member &&
            FileSystem::GetRealFilename("/app01/eboot.bin").empty(),
        "resolve archive mounts at path-component boundaries");
  Check(FileSystem::KernelOpen("/app0/../outside.bin", 0, 0) ==
            Libs::LibKernel::KERNEL_ERROR_ENOENT &&
            FileSystem::KernelOpen("/app0/missing.bin", 0, 0) ==
            Libs::LibKernel::KERNEL_ERROR_ENOENT,
        "reject traversal outside an archive and missing members");

  FileSystem::FileStat path_stat{}, descriptor_stat{};
  Check(FileSystem::KernelStat(GuestMember, &path_stat) == OK &&
            path_stat.st_size == payload.size() &&
            path_stat.st_size == Common::File::Size(host_member) &&
            FileSystem::KernelCheckReachability(GuestMember) == OK,
        "archive path stat and reachability agree with Common::File");
  const int fd = FileSystem::KernelOpen("/app0/ASSETS/subdir/DATA.BIN", 0, 0);
  Check(fd >= 3 && FileSystem::KernelFstat(fd, &descriptor_stat) == OK &&
            descriptor_stat.st_size == path_stat.st_size &&
            descriptor_stat.st_mode == path_stat.st_mode,
        "case-insensitive archive open and descriptor stat agree");
  std::array<uint8_t, 97> bytes{};
  constexpr int64_t Offset = 64 * 1024 - 19;
  Check(FileSystem::KernelPread(fd, bytes.data(), bytes.size(), Offset) == bytes.size() &&
            std::equal(bytes.begin(), bytes.end(), payload.begin() + Offset) &&
            FileSystem::KernelLseek(fd, 0, 1) == 0,
        "archive pread crosses a compression block without moving position");
  Check(FileSystem::KernelLseek(fd, Offset, 0) == Offset &&
            FileSystem::KernelRead(fd, bytes.data(), bytes.size()) == bytes.size() &&
            std::equal(bytes.begin(), bytes.end(), payload.begin() + Offset) &&
            FileSystem::KernelLseek(fd, 0, 1) == Offset + bytes.size(),
        "archive seek and sequential read share descriptor position");
  Check(FileSystem::KernelLseek(fd, -9, 2) == payload.size() - 9 &&
            FileSystem::KernelRead(fd, bytes.data(), bytes.size()) == 9 &&
            std::equal(bytes.begin(), bytes.begin() + 9, payload.end() - 9) &&
            FileSystem::KernelRead(fd, bytes.data(), bytes.size()) == 0,
        "archive reads stop at the member boundary");
  Check(FileSystem::KernelWrite(fd, bytes.data(), 1) ==
            Libs::LibKernel::KERNEL_ERROR_EBADF &&
            FileSystem::KernelPwrite(fd, bytes.data(), 1, 0) ==
            Libs::LibKernel::KERNEL_ERROR_EBADF &&
            FileSystem::KernelFtruncate(fd, 0) ==
            Libs::LibKernel::KERNEL_ERROR_EBADF,
        "archive read-only descriptors reject write and truncate");

  const auto unicode_guest = std::string("/app0/") + std::string(ArchiveTests::UnicodeFilename);
  const int unicode = FileSystem::KernelOpen(unicode_guest.c_str(), 0, 0);
  Check(unicode >= 3 && FileSystem::KernelRead(unicode, bytes.data(), bytes.size()) ==
            ArchiveTests::Eboot.size() &&
            std::memcmp(bytes.data(), ArchiveTests::Eboot.data(), ArchiveTests::Eboot.size()) == 0,
        "open and read a Unicode archive member");
  Check(FileSystem::KernelClose(unicode) == OK, "close Unicode archive member");

  const int directory = FileSystem::KernelOpen("/app0/", 0x00020000, 0);
  Check(directory >= 3, "open mounted archive directory");
  std::array<char, 512> block{};
  const auto expected_entries = Common::File::GetDirEntries(archive_root);
  size_t entries_seen = 0;
  for (;;) {
    const int length = FileSystem::KernelGetdents(directory, block.data(), block.size());
    Check(length >= 0 && length <= block.size(), "read archive directory records");
    if (length == 0) {
      break;
    }
    for (size_t offset = 0; offset < static_cast<size_t>(length);) {
      uint16_t record_length = 0;
      Check(length - offset >= 8, "archive directory record header fits");
      std::memcpy(&record_length, block.data() + offset + 4, sizeof(record_length));
      const auto name_length = static_cast<uint8_t>(block[offset + 7]);
      Check(record_length >= 9 + name_length && record_length <= length - offset,
            "archive directory record fits");
      const std::string_view name(block.data() + offset + 8, name_length);
      Check(std::any_of(expected_entries.begin(), expected_entries.end(), [&](const auto &entry) {
        return entry.name == name && block[offset + 6] == (entry.is_file ? 8 : 4);
      }), "guest directory entry matches Common::File name and type");
      ++entries_seen;
      offset += record_length;
    }
  }
  Check(entries_seen == expected_entries.size(), "guest enumerates every archive directory entry");
  Check(FileSystem::KernelClose(directory) == OK, "close archive directory");

  for (const int flags : {1, 2, 0x0200, 0x0400}) {
    Check(FileSystem::KernelOpen(GuestMember, flags, 0777) ==
              Libs::LibKernel::KERNEL_ERROR_EROFS,
          "archive rejects write, create and truncate open flags");
  }
  Check(FileSystem::KernelUnlink(GuestMember) == Libs::LibKernel::KERNEL_ERROR_EROFS &&
            FileSystem::KernelMkdir("/app0/new-dir", 0777) == Libs::LibKernel::KERNEL_ERROR_EROFS &&
            FileSystem::KernelRmdir("/app0/assets") == Libs::LibKernel::KERNEL_ERROR_EROFS &&
            FileSystem::KernelRename(GuestMember, "/app0/renamed.bin") ==
                Libs::LibKernel::KERNEL_ERROR_EROFS,
        "archive mount rejects path mutations");
  FileSystem::Umount("/app0");
  Check(FileSystem::GetRealFilename(GuestMember).empty() &&
            FileSystem::KernelOpen(GuestMember, 0, 0) == Libs::LibKernel::KERNEL_ERROR_ENOENT &&
            FileSystem::KernelLseek(fd, 0, 0) == 0 &&
            FileSystem::KernelRead(fd, bytes.data(), bytes.size()) == bytes.size() &&
            std::equal(bytes.begin(), bytes.end(), payload.begin()),
        "unmount hides archive paths while open descriptors retain the reader");
  Check(FileSystem::KernelClose(fd) == OK, "close last archive descriptor");
}

void CheckUnicodePaths(const std::filesystem::path &root) {
  constexpr std::string_view HostDirectory =
      "Test\xc3\xa9-\xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e";
  constexpr std::string_view GuestFilename =
      "asset-\xc3\xa9-\xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e.bin";

  const auto unicode_root =
      root / Common::PathFromUtf8(HostDirectory);
  const auto nested_root = unicode_root / "nested";

  Check(Common::File::CreateDirectories(nested_root),
        "create nested Unicode host directory");

  CheckMountRoot(nested_root);

  const auto native_file =
      nested_root / Common::PathFromUtf8(GuestFilename);

  Common::File fixture;
  Check(fixture.Create(native_file), "create Unicode filename");
  fixture.Close();

  const auto entries = Common::File::GetDirEntries(nested_root);
  Check(std::any_of(entries.begin(), entries.end(), [&](const auto &entry) {
          return entry.is_file && entry.name == GuestFilename;
        }), "directory enumeration returns UTF-8 filenames");

  FileSystem::Mount(nested_root, "/app0");

  const auto guest_file =
      std::string("/app0/") + std::string(GuestFilename);

  Check(FileSystem::GetRealFilename(guest_file) == native_file,
        "resolve Unicode guest path");

  const int fd = FileSystem::KernelOpen(guest_file.c_str(), 0, 0);
  Check(fd >= 3, "open Unicode guest path");
  Check(FileSystem::KernelClose(fd) == OK,
        "close Unicode guest path");

  FileSystem::Umount("/app0");
}

void CheckUnicodeLogPath(const std::filesystem::path &root) {
  Config::ConfigOptions options;
  options.printf_direction = Config::LogDirection::File;
  options.printf_output_file = root / u8"logs-\u65e5\u672c\u8a9e-\U0001f600" / u8"log-\u00e9.txt";
  Config::Load(options);
  Log::Initialize();
  constexpr std::string_view Payload = "Unicode log path\n";
  Log::Write(Payload);
  Log::Shutdown();

  Common::File result(options.printf_output_file, Common::File::Mode::Read);
  Check(!result.IsInvalid(), "open Unicode log file");
  const auto data = result.ReadWholeBuffer();
  Check(data.size() == Payload.size() &&
            std::memcmp(data.data(), Payload.data(), data.size()) == 0,
        "Unicode log file contains output");
  options.printf_direction = Config::LogDirection::Silent;
  Config::Load(options);
  Log::Initialize();
}

void CheckDirectoryStream(const std::filesystem::path &root) {
  const auto directory = root / "directory-stream";
  Check(std::filesystem::create_directory(directory),
        "create seek fixture directory");
  for (int i = 0; i < 48; ++i) {
    Common::File fixture;
    Check(fixture.Create(directory / ("directory-entry-" + std::to_string(i))),
          "create enough entries to cross directory blocks");
    fixture.Close();
  }
  FileSystem::Mount(directory, "/app0");
  const int fd = FileSystem::KernelOpen("/app0/", 0, 0);
  Check(fd >= 3, "open directory as read-only asset");
  const auto end = FileSystem::KernelLseek(fd, 0, 2);
  Check(end > 512 && end % 512 == 0,
        "directory SEEK_END uses padded stream size");
  FileSystem::FileStat stat{};
  Check(FileSystem::KernelFstat(fd, &stat) == OK && stat.st_size == end &&
            stat.st_blksize == 512,
        "directory stat agrees with seek and enumeration");
  Check(FileSystem::KernelStat("/app0/", &stat) == OK && stat.st_size == end &&
            stat.st_blocks == end / 512,
        "path and descriptor directory stat agree");
  Check(FileSystem::KernelLseek(fd, 0, 0) == 0,
        "rewind directory for asset read");
  std::vector<char> raw(static_cast<size_t>(end));
  Check(FileSystem::KernelRead(fd, raw.data(), raw.size()) == end &&
            FileSystem::KernelRead(fd, raw.data(), 1) == 0,
        "raw directory read reaches EOF");
  Check(FileSystem::KernelLseek(fd, -end, 1) == 0,
        "directory SEEK_CUR uses the position advanced by read");

  std::array<char, 512> block{};
  int64_t base = -1;
  for (int64_t offset = 0; offset < end; offset += block.size()) {
    Check(FileSystem::KernelGetdirentries(fd, block.data(), block.size(),
                                          &base) == block.size() &&
              base == offset &&
              std::memcmp(block.data(), raw.data() + offset, block.size()) == 0,
          "directory enumeration shares raw bytes and reports each block "
          "position");
    for (size_t pos = 0; pos < block.size();) {
      Check(block.size() - pos >= 8, "directory record header fits");
      uint16_t length = 0;
      std::memcpy(&length, block.data() + pos + 4, sizeof(length));
      const auto name_length = static_cast<uint8_t>(block[pos + 7]);
      Check(length >= 9 + name_length && length <= block.size() - pos &&
                block[pos + 8 + name_length] == '\0',
            "directory entries remain complete within every block");
      pos += length;
    }
  }
  Check(FileSystem::KernelGetdirentries(fd, block.data(), block.size(),
                                        &base) == 0 &&
            base == end,
        "directory enumeration reports EOF position");
  Check(FileSystem::KernelLseek(fd, 512, 0) == 512 &&
            FileSystem::KernelGetdirentries(fd, block.data(), block.size(),
                                            &base) == block.size() &&
            base == 512 &&
            std::memcmp(block.data(), raw.data() + 512, block.size()) == 0,
        "restore and reread a directory enumeration position");

  const auto position = FileSystem::KernelLseek(fd, 0, 1);
  Check(
      FileSystem::KernelLseek(fd, 0, 9) ==
              Libs::LibKernel::KERNEL_ERROR_EINVAL &&
          FileSystem::KernelLseek(fd, -1, 0) ==
              Libs::LibKernel::KERNEL_ERROR_EINVAL &&
          FileSystem::KernelLseek(fd, std::numeric_limits<int64_t>::min(), 1) ==
              Libs::LibKernel::KERNEL_ERROR_EINVAL &&
          FileSystem::KernelLseek(fd, std::numeric_limits<int64_t>::max(), 1) ==
              Libs::LibKernel::KERNEL_ERROR_EOVERFLOW &&
          FileSystem::KernelLseek(fd, 0, 1) == position,
      "invalid and overflowing directory seeks preserve the position");
  Check(FileSystem::KernelLseek(fd, -19, 2) == end - 19 &&
            FileSystem::KernelRead(fd, block.data(), block.size()) == 19 &&
            std::memcmp(block.data(), raw.data() + end - 19, 19) == 0,
        "raw directory reads support byte positions and stop at EOF");
  Check(FileSystem::KernelLseek(fd, 1, 0) == 1 &&
            FileSystem::KernelGetdents(fd, block.data(), block.size()) ==
                Libs::LibKernel::KERNEL_ERROR_EINVAL &&
            FileSystem::KernelLseek(fd, 0, 1) == 1,
        "directory enumeration rejects an incomplete record position");
  Check(FileSystem::KernelLseek(fd, 512, 2) == end + 512 &&
            FileSystem::KernelGetdirentries(fd, block.data(), block.size(),
                                            &base) ==
                Libs::LibKernel::KERNEL_ERROR_EINVAL &&
            FileSystem::KernelLseek(fd, 0, 1) == end + 512,
        "directory enumeration rejects a position beyond EOF");
  Check(FileSystem::KernelRead(
            fd, block.data(),
            static_cast<size_t>(std::numeric_limits<int>::max()) + 1) ==
                Libs::LibKernel::KERNEL_ERROR_EINVAL &&
            FileSystem::KernelLseek(fd, 0, 1) == end + 512,
        "oversized read fails without changing the directory position");
  Check(FileSystem::KernelClose(fd) == OK, "close directory stream");
  Check(FileSystem::KernelLseek(fd, 0, 0) ==
            Libs::LibKernel::KERNEL_ERROR_EBADF,
        "seek rejects a closed descriptor");
  FileSystem::Umount("/app0");
}

void CheckAprPaths(const std::filesystem::path &root) {
  Loader::SymbolDatabase symbols;
  Libs::LibKernelApr::InitLibKernel_1_Apr(&symbols);
  const auto *resolve_symbol = symbols.FindByNid("w5fcCG+t31g", Loader::SymbolType::Func);
  const auto *each_symbol = symbols.FindByNid("C+Khtbbx2g8", Loader::SymbolType::Func);
  Check(resolve_symbol && each_symbol, "APR path exports are registered");
  using Resolve = int (KYTY_SYSV_ABI *)(const char *, const char *const *, uint32_t,
                                      uint32_t *, uint64_t *, uint32_t *);
  using ResolveEach = int (KYTY_SYSV_ABI *)(const char *, const char *const *, uint32_t,
                                          uint32_t *, uint64_t *, int *);
  const auto resolve = reinterpret_cast<Resolve>(resolve_symbol->vaddr);
  const auto resolve_each = reinterpret_cast<ResolveEach>(each_symbol->vaddr);
  Common::File fixture;
  Check(fixture.Create(root / "apr.dat"), "create APR fixture");
  fixture.Write("APR", 3);
  fixture.Close();
  FileSystem::Mount(root, "/app0");

  uint32_t expected_id = 0xffffffffu;
  for (const auto &parts : {std::array{"", "/app0/apr.dat"},
                           std::array{"/app0/", "apr.dat"},
                           std::array{"/", "app0/apr.dat"},
                           std::array{"/app", "0/apr.dat"}}) {
    uint32_t id = 0xffffffffu, error_index = 0xffffffffu;
    uint64_t size = 0;
    Check(resolve(parts[0], &parts[1], 1, &id, &size, &error_index) == OK &&
              id != 0xffffffffu && size == 3,
          "APR concatenates empty, one-character and partial-component prefixes");
    if (expected_id == 0xffffffffu) {
      expected_id = id;
    }
    Check(id == expected_id, "equivalent APR paths return the same ID");
  }

  const char *paths[] = {"/app0/missing.dat", "/app0/apr.dat"};
  uint32_t ids[2] = {}, error_index = 0xffffffffu;
  uint64_t sizes[2] = {1, 1};
  int results[2] = {};
  Check(resolve_each("", paths, 2, ids, sizes, results) == 1 &&
            results[0] == Libs::LibKernel::KERNEL_ERROR_ENOENT && results[1] == OK &&
            ids[0] == 0xffffffffu && ids[1] == expected_id && sizes[0] == 0 && sizes[1] == 3,
        "APR foreach reports a missing path and continues to the valid file");

  // PATH_MAX includes NUL; all components remain below NAME_MAX (255).
  std::string longest = "/app0/";
  for (int i = 0; i < 3; ++i) {
    longest += std::string(254, 'a') + '/';
  }
  longest += std::string(1023 - longest.size(), 'b');
  paths[0] = longest.c_str();
  Check(resolve("", paths, 1, ids, sizes, &error_index) == -1 &&
            *Libs::Posix::GetErrorAddr() == Libs::Posix::POSIX_ENOENT && error_index == 0,
        "APR accepts a pathname whose final NUL is at PATH_MAX minus one");
  Check(resolve("/", paths, 1, ids, sizes, &error_index) == -1 &&
            *Libs::Posix::GetErrorAddr() == Libs::Posix::POSIX_ENAMETOOLONG,
        "APR rejects concatenated paths exceeding PATH_MAX");
  std::array<char, 1024> unterminated;
  unterminated.fill('/');
  paths[0] = unterminated.data();
  Check(resolve("", paths, 1, ids, sizes, &error_index) == -1 &&
            *Libs::Posix::GetErrorAddr() == Libs::Posix::POSIX_ENAMETOOLONG,
        "APR rejects an unterminated pathname");
  paths[0] = "apr.dat";
  Check(resolve(unterminated.data(), paths, 1, ids, sizes, &error_index) == -1 &&
            *Libs::Posix::GetErrorAddr() == Libs::Posix::POSIX_ENAMETOOLONG,
        "APR rejects an unterminated prefix");
  FileSystem::Umount("/app0");
}

void CheckSocketWakeup() {
  namespace Net = Libs::Network::Net;
  Loader::SymbolDatabase symbols;
  Libs::LibNet::InitNet_1_Net(&symbols);
  const auto *send_symbol = symbols.Find(
      {"beRjXBn-z+o", "Net", 1, "Net", 1, 1, Loader::SymbolType::Func});
  const auto *recv_symbol = symbols.Find(
      {"9wO9XrMsNhc", "Net", 1, "Net", 1, 1, Loader::SymbolType::Func});
  const auto *errno_symbol = symbols.Find(
      {"HQOwnfMGipQ", "Net", 1, "Net", 1, 1, Loader::SymbolType::Func});
  Check(send_symbol && recv_symbol && errno_symbol,
        "Net send, receive and errno exports resolve with the guest ABI versions");
  using Send = int (KYTY_SYSV_ABI *)(int, const void *, size_t, int);
  using Recv = int (KYTY_SYSV_ABI *)(int, void *, size_t, int);
  using Errno = int *(KYTY_SYSV_ABI *)();
  const auto net_send = reinterpret_cast<Send>(send_symbol->vaddr);
  const auto net_recv = reinterpret_cast<Recv>(recv_symbol->vaddr);
  auto *net_errno = reinterpret_cast<Errno>(errno_symbol->vaddr)();
  // Guest sockaddr_in: length, family, network-order port/address, padding.
  std::array<uint8_t, 16> address {16, 2, 0, 0, 127, 0, 0, 1};
  const int listener = Net::Socket(2, 1, 0);
  Check(listener >= 0, "create loopback listener");
  Check(Net::Bind(listener, address.data(), address.size()) == 0, "bind loopback");
  Check(Net::Listen(listener, 1) == 0, "listen on loopback");
  uint32_t address_size = address.size();
  Check(Net::Getsockname(listener, address.data(), &address_size) == 0,
        "get assigned loopback port");
  const int writer = Net::Socket(2, 1, 0);
  Check(writer >= 0 && Net::Connect(writer, address.data(), address_size) == 0,
        "connect wake socket");
  const int reader = Net::Accept(listener, nullptr, nullptr);
  Check(reader >= 0, "accept wake socket");
  Check(Net::SocketClose(listener) == 0, "close listener");
  const int enabled = 1;
  Check(Net::Setsockopt(writer, 6, 1, &enabled, sizeof(enabled)) == 0,
        "enable TCP_NODELAY");
  int socket_error = -1;
  uint32_t error_size = sizeof(socket_error);
  *Libs::Posix::GetErrorAddr() = Libs::Posix::POSIX_EINVAL;
  Check(Net::Getsockopt(writer, 0xffff, 0x1007, &socket_error, &error_size) == 0 &&
            socket_error == 0 && error_size == sizeof(socket_error) &&
            *Libs::Posix::GetErrorAddr() == Libs::Posix::POSIX_EINVAL,
        "SO_ERROR reports socket status without changing guest errno");

  std::array<uint64_t, 16> readable {};
  const auto bit = uint64_t {1} << (reader % 64);
  readable[reader / 64] = bit;
  const std::array<int64_t, 2> immediate {0, 0};
  Check(Net::Select(reader + 1, readable.data(), nullptr, nullptr,
                    immediate.data()) == 0 && readable[reader / 64] == 0,
        "empty socket is not readable");
  const char payload[] = "wake";
  *net_errno = Libs::Posix::POSIX_EINVAL;
  Check(net_send(writer, payload, sizeof(payload), 0) == sizeof(payload) &&
            *net_errno == Libs::Posix::POSIX_EINVAL,
        "Net send forwards bytes and preserves errno on success");
  readable[reader / 64] = bit;
  const std::array<int64_t, 2> deadline {1, 0};
  Check(Net::Select(reader + 1, readable.data(), nullptr, nullptr,
                    deadline.data()) == 1 && readable[reader / 64] == bit,
        "select reports the guest descriptor after wake");
  std::array<char, sizeof(payload)> received {};
  Check(net_recv(reader, received.data(), received.size(), 0x42) == sizeof(payload) &&
            std::memcmp(received.data(), payload, sizeof(payload)) == 0 &&
            *net_errno == Libs::Posix::POSIX_EINVAL,
        "Net receive forwards PEEK and WAITALL without consuming bytes");
  received.fill(0);
  Check(net_recv(reader, received.data(), received.size(), 0x40) == sizeof(payload) &&
            std::memcmp(received.data(), payload, sizeof(payload)) == 0,
        "Net receive consumes the same bytes after peeking");
#if !defined(_WIN32)
  Check(Net::Recv(reader, received.data(), received.size(), 0x80) == -1 &&
            *Libs::Posix::GetErrorAddr() == Libs::Posix::POSIX_EWOULDBLOCK,
        "empty nonblocking receive translates guest errno");
  Check(net_recv(reader, received.data(), received.size(), 0x80) ==
            Libs::Network::NET_ERROR_EWOULDBLOCK &&
            *net_errno == Libs::Posix::POSIX_EWOULDBLOCK,
        "Net nonblocking receive translates POSIX failure and Net errno");
#endif
  Check(net_send(-1, payload, sizeof(payload), 0) == Libs::Network::NET_ERROR_EBADF &&
            *net_errno == Libs::Posix::POSIX_EBADF,
        "Net send translates an invalid socket instead of returning POSIX minus one");
  Check(net_recv(reader, nullptr, received.size(), 0) == Libs::Network::NET_ERROR_EFAULT &&
            *net_errno == Libs::Posix::POSIX_EFAULT,
        "Net receive translates an invalid output buffer");
  Check(net_send(writer, payload, sizeof(payload), 0x100000) ==
            Libs::Network::NET_ERROR_EOPNOTSUPP &&
            *net_errno == Libs::Posix::POSIX_EOPNOTSUPP,
        "Net send preserves the backend's unsupported crypto flag error");
#if defined(__linux__)
  const int disconnected = Net::Socket(2, 1, 0);
  Check(disconnected >= 0, "create unconnected socket for broken pipe check");
  const auto previous_sigpipe = std::signal(SIGPIPE, SIG_DFL);
  Check(previous_sigpipe != SIG_ERR, "set default SIGPIPE disposition for Net send");
  const auto broken_send = net_send(disconnected, payload, sizeof(payload), 0);
  std::signal(SIGPIPE, previous_sigpipe);
  Check(broken_send == Libs::Network::NET_ERROR_EPIPE &&
            *net_errno == Libs::Posix::POSIX_EPIPE,
        "Net send reports a broken pipe without raising host SIGPIPE");
  Check(Net::SocketClose(disconnected) == 0, "close unconnected socket");
#endif
  Check(Net::SocketClose(writer) == 0, "close wake writer");
  *net_errno = Libs::Posix::POSIX_EINVAL;
  Check(net_recv(reader, received.data(), received.size(), 0) == 0 &&
            *net_errno == Libs::Posix::POSIX_EINVAL,
        "Net receive returns EOF without replacing errno");
  Check(Net::SocketClose(reader) == 0, "close wake reader");
  readable[reader / 64] = bit;
  Check(Net::Select(reader + 1, readable.data(), nullptr, nullptr,
                    immediate.data()) == -1 &&
            *Libs::Posix::GetErrorAddr() == Libs::Posix::POSIX_EBADF &&
            readable[reader / 64] == bit,
        "closed descriptor fails without clearing input fd_set");
}

#if defined(_WIN32)
namespace Net = Libs::Network::Net;

void SetReceiveTimeout(int socket, int milliseconds) {
  const std::array<int64_t, 2> timeout{milliseconds / 1000,
                                       (milliseconds % 1000) * 1000};
  Check(Net::Setsockopt(socket, 0xffff, 0x1006, timeout.data(),
                        sizeof(timeout)) == 0,
        "set guest timeval receive timeout");
}

void SetSocketNonblocking(int socket, bool enabled) {
  const int value = enabled ? 1 : 0;
  Check(Net::Setsockopt(socket, 0xffff, 0x1200, &value, sizeof(value)) == 0,
        "set guest socket blocking mode");
  int actual = -1;
  uint32_t size = sizeof(actual);
  Check(Net::Getsockopt(socket, 0xffff, 0x1200, &actual, &size) == 0 &&
            actual == value,
        "shared transport reports its blocking mode");
}

void WaitSocketReadable(int socket) {
  std::array<uint64_t, 16> readable{};
  readable[socket / 64] = uint64_t{1} << (socket % 64);
  const std::array<int64_t, 2> timeout{2, 0};
  Check(Net::Select(socket + 1, readable.data(), nullptr, nullptr,
                    timeout.data()) == 1,
        "loopback fixture becomes readable");
}

class LoopbackConnection {
public:
  explicit LoopbackConnection(bool nonblocking_listener = false) {
    std::array<uint8_t, 16> address{16, 2, 0, 0, 127, 0, 0, 1};
    const int listener = Net::Socket(2, 1, 0);
    Check(listener >= 0 &&
              Net::Bind(listener, address.data(), address.size()) == 0 &&
              Net::Listen(listener, 1) == 0,
          "create receive regression listener");
    uint32_t size = address.size();
    Check(Net::Getsockname(listener, address.data(), &size) == 0,
          "get receive regression listener address");
    writer = Net::Socket(2, 1, 0);
    Check(writer >= 0 && Net::Connect(writer, address.data(), size) == 0,
          "connect receive regression writer");
    WaitSocketReadable(listener);
    if (nonblocking_listener) {
      SetSocketNonblocking(listener, true);
    }
    reader = Net::Accept(listener, nullptr, nullptr);
    Check(reader >= 0 && Net::SocketClose(listener) == 0,
          "accept receive regression connection");
    const int enabled = 1;
    Check(Net::Setsockopt(writer, 6, 1, &enabled, sizeof(enabled)) == 0,
          "disable Nagle for gated chunk arrivals");
    SetReceiveTimeout(reader, 2000);
  }

  ~LoopbackConnection() {
    if (reader >= 0) {
      Check(Net::SocketClose(reader) == 0, "close receive regression reader");
    }
    if (writer >= 0) {
      Check(Net::SocketClose(writer) == 0, "close receive regression writer");
    }
  }

  void Send(std::string_view bytes) const {
    Check(Net::Send(writer, bytes.data(), bytes.size(), 0x20000) ==
              bytes.size(),
          "send receive regression bytes");
  }

  KYTY_CLASS_NO_COPY(LoopbackConnection);
  int reader = -1;
  int writer = -1;
};

void CheckWindowsReceiveFlags() {
  using namespace std::chrono_literals;
  constexpr std::string_view payload = "abcdef";

  // The existing wake test covers a prefilled Recv. Exercise Recvfrom as well:
  // Winsock does not fill the source address itself for a stream receive.
  {
    LoopbackConnection connection;
    connection.Send(payload);
    WaitSocketReadable(connection.reader);
    std::array<char, 6> bytes{};
    std::array<uint8_t, 16> peer{}, expected_peer{};
    uint32_t peer_size = peer.size(), expected_size = expected_peer.size();
    Check(Net::Getsockname(connection.writer, expected_peer.data(),
                           &expected_size) == 0,
          "get expected receive peer address");
    Check(Net::Recvfrom(connection.reader, bytes.data(), bytes.size(), 0x42,
                        peer.data(), &peer_size) == bytes.size() &&
              std::string_view(bytes.data(), bytes.size()) == payload &&
              peer == expected_peer,
          "stream Recvfrom PEEK WAITALL preserves bytes and reports peer");
    Check(Net::Recv(connection.reader, bytes.data(), bytes.size(), 0x40) ==
                  bytes.size() &&
              std::string_view(bytes.data(), bytes.size()) == payload,
          "consume bytes after stream Recvfrom peek");
    Check(Net::Recv(connection.reader, bytes.data(), 0, 0x42) == 0,
          "zero-length stream PEEK WAITALL returns immediately");
  }

  for (const int flags : {0x40, 0x42}) {
    LoopbackConnection connection;
    connection.Send(payload.substr(0, 2));
    WaitSocketReadable(connection.reader);
    std::array<char, 6> bytes{};
    std::promise<void> entered;
    auto entry = entered.get_future();
    auto receive = std::async(std::launch::async, [&] {
      entered.set_value();
      return Net::Recv(connection.reader, bytes.data(), bytes.size(), flags);
    });
    Check(entry.wait_for(2s) == std::future_status::ready,
          "start chunked receive");
    Check(receive.wait_for(50ms) == std::future_status::timeout,
          "WAITALL remains pending with only the first queued chunk");
    connection.Send(payload.substr(2, 2));
    Check(receive.wait_for(50ms) == std::future_status::timeout,
          "WAITALL remains pending with only two chunks");
    connection.Send(payload.substr(4));
    Check(receive.wait_for(2s) == std::future_status::ready &&
              receive.get() == bytes.size() &&
              std::string_view(bytes.data(), bytes.size()) == payload,
          "WAITALL returns all three gated chunks in order");
    if ((flags & 2) != 0) {
      bytes.fill(0);
      Check(Net::Recv(connection.reader, bytes.data(), bytes.size(), 0x40) ==
                    bytes.size() &&
                std::string_view(bytes.data(), bytes.size()) == payload,
            "chunked PEEK leaves the complete stream queued");
    }
  }

  {
    LoopbackConnection connection(true);
    std::array<char, 6> bytes{};
    int inherited = 0;
    uint32_t size = sizeof(inherited);
    Check(Net::Getsockopt(connection.reader, 0xffff, 0x1200, &inherited,
                          &size) == 0 &&
              inherited == 1,
          "accepted Windows socket inherits nonblocking mode");
    for (const int flags : {0x40, 0x42, 0x80, 0xc2}) {
      Check(
          Net::Recv(connection.reader, bytes.data(), bytes.size(), flags) ==
                  -1 &&
              *Libs::Posix::GetErrorAddr() == Libs::Posix::POSIX_EWOULDBLOCK,
          "empty nonblocking receive ignores WAITALL and reports would-block");
    }
    connection.Send(payload.substr(0, 2));
    WaitSocketReadable(connection.reader);
    Check(Net::Recv(connection.reader, bytes.data(), bytes.size(), 0xc2) == 2 &&
              std::string_view(bytes.data(), 2) == payload.substr(0, 2),
          "nonblocking PEEK WAITALL returns the queued prefix");
    Check(Net::Recv(connection.reader, bytes.data(), bytes.size(), 0x40) == 2 &&
              std::string_view(bytes.data(), 2) == payload.substr(0, 2),
          "nonblocking WAITALL consumes only the queued prefix");
    SetSocketNonblocking(connection.reader, false);
    Check(
        Net::Recv(connection.reader, bytes.data(), bytes.size(), 0x80) == -1 &&
            *Libs::Posix::GetErrorAddr() == Libs::Posix::POSIX_EOPNOTSUPP,
        "blocking transport explicitly rejects unsupported per-call DONTWAIT");
  }

  for (const int flags : {0x40, 0x42}) {
    LoopbackConnection connection;
    SetReceiveTimeout(connection.reader, 80);
    std::array<int64_t, 2> timeout{};
    uint32_t size = sizeof(timeout);
    Check(Net::Getsockopt(connection.reader, 0xffff, 0x1006, timeout.data(),
                          &size) == 0 &&
              timeout == std::array<int64_t, 2>{0, 80'000},
          "receive timeout round-trips through the guest timeval ABI");
    const std::array<int64_t, 2> invalid_timeout{0, 1'000'000};
    Check(Net::Setsockopt(connection.reader, 0xffff, 0x1006,
                          invalid_timeout.data(),
                          sizeof(invalid_timeout)) == -1 &&
              *Libs::Posix::GetErrorAddr() == Libs::Posix::POSIX_EINVAL,
          "invalid timeval does not change the receive timeout");
    std::array<char, 6> bytes{};
    Check(Net::Recv(connection.reader, bytes.data(), bytes.size(), flags) ==
                  -1 &&
              *Libs::Posix::GetErrorAddr() == Libs::Posix::POSIX_EWOULDBLOCK,
          "empty WAITALL receive expires with would-block");
    connection.Send(payload.substr(0, 2));
    WaitSocketReadable(connection.reader);
    *Libs::Posix::GetErrorAddr() = Libs::Posix::POSIX_EINVAL;
    const auto start = std::chrono::steady_clock::now();
    Check(Net::Recv(connection.reader, bytes.data(), bytes.size(), flags) ==
                  2 &&
              std::string_view(bytes.data(), 2) == payload.substr(0, 2) &&
              *Libs::Posix::GetErrorAddr() == Libs::Posix::POSIX_EINVAL,
          "WAITALL timeout returns the prefix without replacing guest errno");
    Check(std::chrono::steady_clock::now() - start >= 70ms,
          "partial WAITALL receive honors the configured deadline");
    if ((flags & 2) != 0) {
      Check(Net::Recv(connection.reader, bytes.data(), 2, 0) == 2 &&
                std::string_view(bytes.data(), 2) == payload.substr(0, 2),
            "timed-out PEEK leaves its prefix available to consume");
    }
    SetSocketNonblocking(connection.reader, true);
    Check(Net::Recv(connection.reader, bytes.data(), bytes.size(), 0) == -1 &&
              *Libs::Posix::GetErrorAddr() == Libs::Posix::POSIX_EWOULDBLOCK,
          "timeout path does not duplicate consumed bytes");
  }

  for (const int flags : {0x40, 0x42}) {
    LoopbackConnection connection;
    SetReceiveTimeout(connection.reader, 0);
    connection.Send(payload.substr(0, 2));
    Check(Net::Shutdown(connection.writer, 1) == 0,
          "send FIN after a partial payload");
    std::array<char, 6> bytes{};
    auto receive = std::async(std::launch::async, [&] {
      return Net::Recv(connection.reader, bytes.data(), bytes.size(), flags);
    });
    Check(receive.wait_for(2s) == std::future_status::ready &&
              receive.get() == 2 &&
              std::string_view(bytes.data(), 2) == payload.substr(0, 2),
          "partial WAITALL completes on EOF without a receive timeout");
    if ((flags & 2) != 0) {
      Check(
          Net::Recv(connection.reader, bytes.data(), bytes.size(), 0x42) == 2 &&
              Net::Recv(connection.reader, bytes.data(), bytes.size(), 0x40) ==
                  2,
          "EOF PEEK preserves the prefix for another peek and consumption");
    }
    Check(Net::Recv(connection.reader, bytes.data(), bytes.size(), flags) == 0,
          "drained stream reports EOF with WAITALL flags");
  }

  for (const int flags : {0x40, 0x42}) {
    LoopbackConnection connection;
    connection.Send(payload.substr(0, 2));
    WaitSocketReadable(connection.reader);
    std::array<char, 6> bytes{};
    std::promise<void> entered;
    auto entry = entered.get_future();
    auto receive = std::async(std::launch::async, [&] {
      *Libs::Posix::GetErrorAddr() = Libs::Posix::POSIX_EINVAL;
      entered.set_value();
      const auto result =
          Net::Recv(connection.reader, bytes.data(), bytes.size(), flags);
      return std::pair{result, *Libs::Posix::GetErrorAddr()};
    });
    Check(entry.wait_for(2s) == std::future_status::ready &&
              receive.wait_for(50ms) == std::future_status::timeout,
          "partial receive is waiting before peer reset");
    // This Windows backend also accepts the native SO_LINGER representation.
    const std::array<uint16_t, 2> abortive_linger{1, 0};
    Check(Net::Setsockopt(connection.writer, 0xffff, 0x80,
                          abortive_linger.data(),
                          sizeof(abortive_linger)) == 0 &&
              Net::SocketClose(connection.writer) == 0,
          "reset peer after partial receive");
    connection.writer = -1;
    Check(receive.wait_for(2s) == std::future_status::ready,
          "peer reset releases partial WAITALL");
    const auto result = receive.get();
    Check(result.first == 2 && result.second == Libs::Posix::POSIX_EINVAL &&
              std::string_view(bytes.data(), 2) == payload.substr(0, 2),
          "peer reset returns already received bytes without replacing errno");
  }

  {
    std::array<char, 6> bytes{};
    const int unconnected = Net::Socket(2, 1, 0);
    Check(unconnected >= 0, "create unconnected receive fixture");
    SetReceiveTimeout(unconnected, 100);
    Check(Net::Recv(unconnected, bytes.data(), bytes.size(), 0x42) == -1 &&
              *Libs::Posix::GetErrorAddr() == Libs::Posix::POSIX_ENOTCONN,
          "WAITALL preserves unconnected stream error");
    Check(Net::SocketClose(unconnected) == 0,
          "close unconnected receive fixture");
    LoopbackConnection connection;
    SetReceiveTimeout(connection.reader, 0);
    std::promise<void> entered;
    auto entry = entered.get_future();
    auto receive = std::async(std::launch::async, [&] {
      entered.set_value();
      const auto result =
          Net::Recv(connection.reader, bytes.data(), bytes.size(), 0x42);
      return std::pair{result, *Libs::Posix::GetErrorAddr()};
    });
    Check(entry.wait_for(2s) == std::future_status::ready &&
              receive.wait_for(50ms) == std::future_status::timeout,
          "empty WAITALL is pending before local shutdown");
    Check(Net::Shutdown(connection.reader, 0) == 0,
          "shut down a pending receive");
    Check(receive.wait_for(2s) == std::future_status::ready,
          "local shutdown cancels WAITALL without a receive timeout");
    const auto result = receive.get();
    Check(result.first == -1 && result.second == Libs::Posix::POSIX_ESHUTDOWN,
          "WAITALL preserves local receive-shutdown error");
  }
}
#endif

} // namespace

int main(int, char**) {
  Common::InitializeThreads();
  Common::Subsystems subsystems;
  subsystems.Initialize<Config::Lifecycle>();
  Config::ConfigOptions options;
  options.printf_direction = Config::LogDirection::Silent;
  Config::Load(options);
  subsystems.Initialize<Log::Lifecycle>();

  Check(SDL_InitSubSystem(SDL_INIT_VIDEO), "initialize Vulkan test video");
  auto graphics = std::make_unique<Libs::Graphics::WindowContext>();
  graphics->graphic_ctx.screen_width = 64;
  graphics->graphic_ctx.screen_height = 64;
  graphics->window = SDL_CreateWindow("KernelFileSystemTests", 64, 64,
                                      SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN);
  Check(graphics->window != nullptr, "create hidden Vulkan test window");
  graphics->CreateVulkan();

  TempDirectory temporary;
  FileSystem::Initialize();
  CheckMountRoot(temporary.Path());
  CheckUnmappedPaths(temporary.Path());
  CheckArchiveMount(temporary.Path());
  CheckUnicodePaths(temporary.Path());
  CheckUnicodeLogPath(temporary.Path());
  CheckDirectoryStream(temporary.Path());
  CheckAprPaths(temporary.Path());
  FileSystem::Mount(temporary.Path(), "/savedata0");
  TestSaveOpenVisibility();
  CheckSaveRename(temporary.Path(), "first-save");
  CheckSaveRename(temporary.Path(), "replacement-save");
  FileSystem::Shutdown();
  CheckSocketWakeup();
#if defined(_WIN32)
  CheckWindowsReceiveFlags();
#endif
  graphics.reset();
  subsystems.Destroy();

  std::printf("KernelFileSystemTests: all cases passed\n");
  return 0;
}
