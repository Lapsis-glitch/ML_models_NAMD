#include "artifact_bundle.h"
#include "pjrt_plugin.h"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Args {
    std::string plugin = "/home/rat/miniconda3/envs/fennix/lib/python3.11/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so";
    std::string manifest = "/home/rat/PycharmProjects/ML_models_NAMD/models/fennix_bio1_stablehlo_n3/manifest.json";
    std::string stablehlo;
    std::string compile_options;
    std::string reference;
    std::string coords;
    bool skip_validate = false;
};

void usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " [--plugin PATH] [--manifest PATH] [--stablehlo PATH] [--compile-options PATH] [--reference PATH] [--coords PATH] [--skip-validate]\n";
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
        else if (key == "--coords") args.coords = need_value("--coords");
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
    std::istringstream in(read_text_file(path));
    std::string line;
    while (std::getline(in, line)) {
        if (line.find(needle) != std::string::npos) return line;
    }
    return "";
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

size_t num_elements(const std::vector<int64_t>& dims) {
    size_t total = 1;
    for (int64_t dim : dims) {
        if (dim <= 0) throw std::runtime_error("dims must be positive");
        total *= static_cast<size_t>(dim);
    }
    return total;
}

bool same_values(const std::vector<float>& a, const std::vector<float>& b, double atol = 1e-7) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(static_cast<double>(a[i] - b[i])) > atol) return false;
    }
    return true;
}

void require_dims(const std::vector<int64_t>& actual, const std::vector<int64_t>& expected, const char* label) {
    if (actual != expected) {
        throw std::runtime_error(std::string(label) + " dims mismatch: got " + dims_to_string(actual) +
                                 " expected " + dims_to_string(expected));
    }
}

std::vector<float> scaled_values(const std::vector<float>& values, double factor) {
    std::vector<float> out(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        out[i] = static_cast<float>(static_cast<double>(values[i]) * factor);
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        ArtifactSpec spec = load_artifact_spec(args.manifest);
        if (!args.stablehlo.empty()) spec.stablehlo_path = args.stablehlo;
        if (!args.compile_options.empty()) spec.compile_options_path = args.compile_options;
        if (!args.reference.empty()) spec.reference_runtime_path = args.reference;
        const RuntimeReference runtime_ref = load_runtime_reference(spec.reference_runtime_path);
        require_dims(runtime_ref.input_dims, spec.input_dims, "manifest/runtime-reference input");

        const std::vector<float> input_values = args.coords.empty()
            ? runtime_ref.coordinates
            : load_flat_float_values(args.coords);
        if (input_values.size() != num_elements(spec.input_dims)) {
            throw std::runtime_error("input coordinate count does not match manifest input dims " + dims_to_string(spec.input_dims));
        }
        const bool validating_reference_coords = same_values(input_values, runtime_ref.coordinates);

        std::cout << "=== FENNIX PJRT native probe ===\n";
        std::cout << "plugin:   " << file_summary(args.plugin) << "\n";
        std::cout << "manifest: " << file_summary(spec.manifest_path) << "\n";
        std::cout << "stablehlo:" << file_summary(spec.stablehlo_path) << "\n";
        std::cout << "compile:  " << file_summary(spec.compile_options_path) << "\n";
        std::cout << "reference:" << file_summary(spec.reference_runtime_path) << "\n";
        if (!args.coords.empty()) {
            std::cout << "coords:   " << file_summary(args.coords) << "\n";
        }

        std::string signature = first_line_containing(spec.stablehlo_path, "func.func public @main");
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

        const std::string mlir = read_text_file(spec.stablehlo_path);
        const std::string compile_options = read_text_file(spec.compile_options_path);

        std::cout << "\nCompiling StableHLO...\n";
        PjrtExecutableHandle executable = plugin.compile_mlir(mlir, compile_options);
        PjrtExecutionResult result;
        try {
            std::cout << "Executable: " << executable.executable_name << "\n";
            std::cout << "Warming up compiled executable once before validation...\n";
            auto warmup = plugin.execute_compiled(executable, input_values, spec.input_dims);
            (void)warmup;
            std::cout << "Running measured execution...\n";
            result = plugin.execute_compiled(executable, input_values, spec.input_dims);
            plugin.destroy_compiled(executable);
        } catch (...) {
            plugin.destroy_compiled(executable);
            throw;
        }
        std::cout << "Input coords (Angstrom): " << values_to_string(input_values) << "\n";

        for (size_t i = 0; i < result.outputs.size(); ++i) {
            const auto& output = result.outputs[i];
            std::cout << "Output[" << i << "] type=" << buffer_type_to_string(output.type)
                      << " dims=" << dims_to_string(output.dims)
                      << " values=" << values_to_string(output.f32_values) << "\n";
        }

        if (result.outputs.size() != 2) {
            throw std::runtime_error("expected exactly 2 outputs for the exported FENNIX artifact");
        }
        require_dims(result.outputs[0].dims, spec.energy_dims, "energy output");
        require_dims(result.outputs[1].dims, spec.forces_dims, "forces output");
        std::cout << "Energy [kcal/mol]:        " << values_to_string(scaled_values(result.outputs[0].f32_values, spec.ev_to_kcal)) << "\n";
        std::cout << "Forces [kcal/mol/A]:      " << values_to_string(scaled_values(result.outputs[1].f32_values, spec.ev_to_kcal)) << "\n";

        if (!args.skip_validate) {
            if (!validating_reference_coords) {
                std::cout << "Validation skipped: custom coordinates differ from the exported runtime reference.\n";
            } else {
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
