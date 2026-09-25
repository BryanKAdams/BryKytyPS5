#include "common/settingsFile.h"

#include <algorithm>
#include <cstdio>
#include <mutex>

namespace Common::SettingsFile {

namespace {

std::mutex g_file_mutex;

std::string_view Trim(std::string_view text) {
	constexpr std::string_view Blank = " \t\r\n";
	const auto                 first = text.find_first_not_of(Blank);
	if (first == std::string_view::npos) {
		return {};
	}
	text = text.substr(first, text.find_last_not_of(Blank) - first + 1);
	if (text.size() >= 2 && text.front() == '"' && text.back() == '"') {
		text = text.substr(1, text.size() - 2);
	}
	return text;
}

bool ReadFile(std::string& text) {
	std::FILE* file = std::fopen(FileName, "rb");
	if (file == nullptr) {
		return false;
	}
	char chunk[4096];
	for (size_t read = 0; (read = std::fread(chunk, 1, sizeof(chunk), file)) != 0;) {
		text.append(chunk, read);
	}
	std::fclose(file);
	return true;
}

std::vector<std::string_view> SplitLines(std::string_view text) {
	std::vector<std::string_view> lines;
	for (size_t start = 0; start < text.size();) {
		const auto end = std::min(text.find('\n', start), text.size());
		lines.push_back(text.substr(start, end - start));
		start = end + 1;
	}
	return lines;
}

// The option a line names, or an empty view for blank and comment lines.
std::string_view LineName(std::string_view line) {
	line = line.substr(0, line.find('#'));
	return Trim(line.substr(0, line.find('=')));
}

} // namespace

bool LoadArguments(std::vector<std::string>& args) {
	std::string text;
	{
		std::scoped_lock lock(g_file_mutex);
		if (!ReadFile(text)) {
			return true;
		}
	}
	size_t line_number = 0;
	for (auto line: SplitLines(text)) {
		line_number++;
		line = line.substr(0, line.find('#'));
		if (Trim(line).empty()) {
			continue;
		}
		const auto separator = line.find('=');
		const auto name      = Trim(line.substr(0, separator));
		if (name.empty() || name.find_first_of(" \t") != std::string_view::npos) {
			std::printf("%s:%zu: expected \"name = value\"\n", FileName, line_number);
			return false;
		}
		args.push_back("--" + std::string(name));
		if (separator != std::string_view::npos) {
			if (const auto value = Trim(line.substr(separator + 1)); !value.empty()) {
				args.emplace_back(value);
			}
		}
	}
	return true;
}

bool Save(std::string_view name, std::string_view value) {
	std::scoped_lock lock(g_file_mutex);
	std::string      text;
	(void)ReadFile(text);
	const std::string setting = std::string(name) + " = " + std::string(value);
	std::string       output;
	bool              replaced = false;
	for (const auto line: SplitLines(text)) {
		if (!replaced && LineName(line) == name) {
			output += setting;
			output += line.ends_with('\r') ? "\r\n" : "\n";
			replaced = true;
			continue;
		}
		output.append(line);
		output += '\n';
	}
	if (!replaced) {
		output += setting;
		output += '\n';
	}
	std::FILE* file = std::fopen(FileName, "wb");
	if (file == nullptr) {
		return false;
	}
	const bool written = std::fwrite(output.data(), 1, output.size(), file) == output.size();
	return std::fclose(file) == 0 && written;
}

} // namespace Common::SettingsFile
