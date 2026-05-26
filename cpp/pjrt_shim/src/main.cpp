#include "pjrt_plugin.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Args {
    std::string plugin = "/home/rat/miniconda3/envs/fennix/lib/python3.11/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so";
    std::string manifest = "/home/rat/PycharmProjects/ML_models_NAMD/models/fennix_bio1_stablehlo_n3/manifest.json";
    std::string stablehlo = "/home/rat/PycharmProjects/ML_models_NAMD/models/fennix_bio1_stablehlo_n3/fennix_bio1_eval.stablehlo.mlir";
    std::string compile_options = "/home/rat/PycharmProjects/ML_models_NAMD/models/fennix_bio1_stablehlo_n3/compile_options.pb";
    std::string reference = "/home/rat/PycharmProjects/ML_models_NAMD/models/fennix_bio1_stablehlo_n3/reference_runtime.txt";
    bool skip_validate = false;
};

void usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " [--plugin PATH] [--manifest PATH] [--stablehlo PATH] [--compile-options PATH] [--reference PATH] [--skip-validate]\n";
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto need_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string(name) + " requires a value");
            }
            return argv[++i];
        };
        if (key == "--plugin") args.plugin = need_value("--plugin");
        else if (key == "--manifest") args.manifest = need_value("--manifest");
        else if (key == "--stablehlo") args.stablehlo = need_value("--stablehlo");
        else if (key == "--compile-options") args.compile_options = need_value("--compile-options");
        else if (key == "--reference") args.reference = need_value("--reference");
        else if (key == "--skip-validate") args.skip_validate = true;
        else if (key == "--help" || key == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + key);
        }
    }
    return args;
}

std::string file_summary(const std::string& path) {
    std::ostringstream os;
    os << path;
    if (!fs::exists(path)) {
        os << " (missing)";
        return os.str();
    }
    os << " (" << fs::file_size(path) << " bytes)";
    return os.str();
}

std::string first_line_containing(const std::string& path, const std::string& needle) {
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.find(needle) != std::string::npos) return line;
    }
    return "";
}

std::string read_file_contents(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open file: " + path);
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

std::vector<float> default_reference_coords() {
    return {
        0.0f, 0.0f, 0.0f,
        0.9572f, 0.0f, 0.0f,
        -0.239987f, 0.927297f, 0.0f,
    };
}

struct RuntimeReference {
    std::vector<int64_t> input_dims;
    std::vector<float> coordinates;
    std::vector<float> energy_ref_ev;
    std::vector<float> forces_ref_ev_a;
    std::vector<float> energy_jit_ev;
    std::vector<float> forces_jit_ev_a;
    double tol_energy_ev = 0.0;
    double tol_forces_ev_a = 0.0;
};

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
    for (const auto& item : split_csv(text)) out.push_back(std::stof(item));
    return out;
}

std::vector<int64_t> parse_int64_csv(const std::string& text) {
    std::vector<int64_t> out;
    if (text.empty()) return out;
    for (const auto& item : split_csv(text)) out.push_back(std::stoll(item));
    return out;
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

std::string dims_to_string(const std::vector<int64_t>& dims) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i) os << ",";
        os << dims[i];
    }
    os << "]";
    return os.str();
}

std::string values_to_string(const std::vector<float>& values) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(8) << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) os << ", ";
        os << values[i];
    }
    os << "]";
    return os.str();
}

std::string buffer_type_to_string(PJRT_Buffer_Type type) {
    switch (type) {
        case PJRT_Buffer_Type_F32: return "F32";
        case PJRT_Buffer_Type_F64: return "F64";
        case PJRT_Buffer_Type_S32: return "S32";
        case PJRT_Buffer_Type_S64: return "S64";
        default: return "<unsupported>";
    }
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) throw std::runtime_error("cannot compare vectors of different sizes");
    double best = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        best = std::max(best, static_cast<double>(std::abs(a[i] - b[i])));
    }
    return best;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);

        std::cout << "=== FENNIX PJRT native probe ===\n";
        std::cout << "plugin:   " << file_summary(args.plugin) << "\n";
        std::cout << "manifest: " << file_summary(args.manifest) << "\n";
        std::cout << "stablehlo:" << file_summary(args.stablehlo) << "\n";
        std::cout << "compile:  " << file_summary(args.compile_options) << "\n";
        std::cout << "reference:" << file_summary(args.reference) << "\n";

        std::string signature = first_line_containing(args.stablehlo, "func.func public @main");
        if (!signature.empty()) {
            std::cout << "StableHLO signature: " << signature << "\n";
        }

        PjrtPlugin plugin(args.plugin);
        plugin.load();
        std::cout << "PJRT API version: " << plugin.api_version_string() << "\n";

        std::cout << "Plugin attributes:\n";
        for (const auto& attr : plugin.plugin_attributes()) {
            std::cout << "  " << attr << "\n";
        }

        plugin.initialize();
        std::cout << "Plugin initialized.\n";

        plugin.create_client();
        std::cout << "Client created.\n";
        std::cout << "Platform: " << plugin.platform_name() << "\n";
        std::cout << "Platform version: " << plugin.platform_version() << "\n";

        std::cout << "Addressable devices:\n";
        auto devices = plugin.device_strings(true);
        if (devices.empty()) std::cout << "  <none>\n";
        for (const auto& d : devices) std::cout << "  " << d << "\n";

        const RuntimeReference runtime_ref = load_runtime_reference(args.reference);
        const std::vector<int64_t> input_dims = runtime_ref.input_dims;
        const std::vector<float> input_values = runtime_ref.coordinates.empty() ? default_reference_coords() : runtime_ref.coordinates;
        const std::string mlir = read_file_contents(args.stablehlo);
        const std::string compile_options = read_file_contents(args.compile_options);

        std::cout << "\nCompiling and executing StableHLO...\n";
        auto result = plugin.compile_and_execute_mlir(mlir, compile_options, input_values, input_dims);
        std::cout << "Executable: " << result.executable_name << "\n";
        std::cout << "Input coords (Angstrom): " << values_to_string(input_values) << "\n";

        for (size_t i = 0; i < result.outputs.size(); ++i) {
            const auto& output = result.outputs[i];
            std::cout << "Output[" << i << "] type=" << buffer_type_to_string(output.type)
                      << " dims=" << dims_to_string(output.dims)
                      << " values=" << values_to_string(output.f32_values) << "\n";
        }

        if (!args.skip_validate) {
            if (result.outputs.size() != 2) {
                throw std::runtime_error("expected exactly 2 outputs for the exported FENNIX artifact");
            }
            const double energy_diff = max_abs_diff(result.outputs[0].f32_values, runtime_ref.energy_jit_ev);
            const double force_diff = max_abs_diff(result.outputs[1].f32_values, runtime_ref.forces_jit_ev_a);

            std::cout << std::scientific << std::setprecision(8);
            std::cout << "Validation vs exported JIT reference:\n";
            std::cout << "  energy max abs diff [eV]      = " << energy_diff
                      << " (tol " << runtime_ref.tol_energy_ev << ")\n";
            std::cout << "  forces max abs diff [eV/A]    = " << force_diff
                      << " (tol " << runtime_ref.tol_forces_ev_a << ")\n";

            if (energy_diff > runtime_ref.tol_energy_ev || force_diff > runtime_ref.tol_forces_ev_a) {
                throw std::runtime_error("native PJRT execution exceeded exported reference tolerances");
            }
        } else {
            std::cout << "Validation skipped by request (--skip-validate).\n";
        }

        std::cout << "\nSTATUS: native PJRT compile/execute probe succeeded.\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "ERROR: " << exc.what() << "\n";
        return 1;
    }
}
