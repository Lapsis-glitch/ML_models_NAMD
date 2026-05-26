#include "artifact_bundle.h"

#include <cctype>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace {

std::vector<std::string> split_csv(const std::string& text) {
	std::vector<std::string> out;
	std::stringstream ss(text);
	std::string item;
	while (std::getline(ss, item, ',')) out.push_back(item);
	return out;
}

std::vector<float> parse_float_csv(const std::string& text) {
	std::vector<float> out;
	if (text.empty()) return out;
	for (const auto& item : split_csv(text)) {
		if (!item.empty()) out.push_back(std::stof(item));
	}
	return out;
}

std::vector<int64_t> parse_int64_csv(const std::string& text) {
	std::vector<int64_t> out;
	if (text.empty()) return out;
	for (const auto& item : split_csv(text)) {
		if (!item.empty()) out.push_back(std::stoll(item));
	}
	return out;
}

std::string extract_match(const std::string& text, const std::regex& re, const char* label) {
	std::smatch match;
	if (!std::regex_search(text, match, re) || match.size() < 2) {
		throw std::runtime_error(std::string("failed to extract ") + label + " from manifest");
	}
	return match[1].str();
}

std::string extract_string(const std::string& text, const std::string& key) {
	return extract_match(text, std::regex("\"" + key + "\"\\s*:\\s*\"([^\"]+)\""), key.c_str());
}

double extract_number(const std::string& text, const std::string& key) {
	return std::stod(extract_match(text, std::regex("\"" + key + "\"\\s*:\\s*([-+0-9.eE]+)"), key.c_str()));
}

std::vector<int64_t> extract_shape_for_name(const std::string& text, const std::string& name) {
	const std::string needle = std::string("\"name\"\\s*:\\s*\"") + name + "\"";
	const std::regex re(needle + R"([\s\S]*?"shape"\s*:\s*\[([^\]]*)\])");
	return parse_int64_csv(extract_match(text, re, name.c_str()));
}

std::vector<float> extract_numeric_values(const std::string& text) {
	static const std::regex number_re(R"([-+]?(?:\d+\.\d*|\d+|\.\d+)(?:[eE][-+]?\d+)?)");
	std::vector<float> values;
	for (auto it = std::sregex_iterator(text.begin(), text.end(), number_re);
		 it != std::sregex_iterator(); ++it) {
		values.push_back(std::stof((*it).str()));
	}
	return values;
}

}  // namespace

std::string read_text_file(const std::string& path) {
	std::ifstream in(path, std::ios::binary);
	if (!in) throw std::runtime_error("failed to open file: " + path);
	std::ostringstream buffer;
	buffer << in.rdbuf();
	return buffer.str();
}

ArtifactSpec load_artifact_spec(const std::string& manifest_path) {
	const fs::path manifest = fs::weakly_canonical(fs::path(manifest_path));
	const std::string text = read_text_file(manifest.string());
	ArtifactSpec spec;
	spec.manifest_path = manifest.string();
	spec.input_dims = extract_shape_for_name(text, "coordinates");
	spec.energy_dims = extract_shape_for_name(text, "energy");
	spec.forces_dims = extract_shape_for_name(text, "forces");
	spec.ev_to_kcal = extract_number(text, "ev_to_kcal");

	const fs::path base = manifest.parent_path();
	spec.stablehlo_path = fs::weakly_canonical(base / extract_string(text, "stablehlo_mlir")).string();
	spec.compile_options_path = fs::weakly_canonical(base / extract_string(text, "compile_options_pb")).string();
	spec.reference_runtime_path = fs::weakly_canonical(base / extract_string(text, "reference_runtime")).string();
	return spec;
}

RuntimeReference load_runtime_reference(const std::string& path) {
	std::ifstream in(path);
	if (!in) throw std::runtime_error("failed to open runtime reference: " + path);
	RuntimeReference ref;
	std::string line;
	while (std::getline(in, line)) {
		if (line.empty()) continue;
		const auto pos = line.find('=');
		if (pos == std::string::npos) throw std::runtime_error("malformed runtime reference line: " + line);
		const std::string key = line.substr(0, pos);
		const std::string value = line.substr(pos + 1);
		if (key == "input_dims") ref.input_dims = parse_int64_csv(value);
		else if (key == "coordinates") ref.coordinates = parse_float_csv(value);
		else if (key == "energy_ref_ev") ref.energy_ref_ev = parse_float_csv(value);
		else if (key == "forces_ref_ev_a") ref.forces_ref_ev_a = parse_float_csv(value);
		else if (key == "energy_jit_ev") ref.energy_jit_ev = parse_float_csv(value);
		else if (key == "forces_jit_ev_a") ref.forces_jit_ev_a = parse_float_csv(value);
		else if (key == "tol_energy_ev") ref.tol_energy_ev = std::stod(value);
		else if (key == "tol_forces_ev_a") ref.tol_forces_ev_a = std::stod(value);
	}
	if (ref.input_dims.empty() || ref.coordinates.empty() || ref.energy_jit_ev.empty() || ref.forces_jit_ev_a.empty()) {
		throw std::runtime_error("runtime reference is missing required fields");
	}
	return ref;
}

std::vector<float> load_flat_float_values(const std::string& path) {
	std::istringstream in(read_text_file(path));
	std::ostringstream filtered;
	std::string line;
	while (std::getline(in, line)) {
		const auto comment = line.find('#');
		filtered << line.substr(0, comment) << '\n';
	}
	const std::vector<float> values = extract_numeric_values(filtered.str());
	if (values.empty()) throw std::runtime_error("no numeric values found in file: " + path);
	return values;
}

