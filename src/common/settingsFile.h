#ifndef EMULATOR_SRC_COMMON_SETTINGSFILE_H_
#define EMULATOR_SRC_COMMON_SETTINGSFILE_H_

#include <string>
#include <string_view>
#include <vector>

// kyty_settings.ini in the working directory holds default options, one per line, named like the
// command-line flags without "--" (e.g. "gpu-timestamp-scale = 115"). '#' starts a comment, and a
// switch like "fullscreen" needs no value. The in-game settings panel (F2) saves its changes here.
namespace Common::SettingsFile {

constexpr const char* FileName = "kyty_settings.ini";

// Appends the file's options as command-line arguments ("--name", "value"). A missing file adds
// nothing. Returns false after printing the line number of a malformed line.
bool LoadArguments(std::vector<std::string>& args);

// Sets "name = value": replaces the first line naming the option, or appends a new line.
bool Save(std::string_view name, std::string_view value);

} // namespace Common::SettingsFile

#endif // EMULATOR_SRC_COMMON_SETTINGSFILE_H_
