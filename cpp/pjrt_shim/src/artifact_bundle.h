#pragma once

#include <cstdint>
#include <string>
#include <vector>

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

struct ArtifactSpec {
    std::string manifest_path;
    std::string stablehlo_path;
    std::string compile_options_path;
    std::string reference_runtime_path;
    std::vector<int64_t> input_dims;
    std::vector<int64_t> energy_dims;
    std::vector<int64_t> forces_dims;
    double ev_to_kcal = 1.0;
};

ArtifactSpec load_artifact_spec(const std::string& manifest_path);
RuntimeReference load_runtime_reference(const std::string& path);
std::string read_text_file(const std::string& path);
std::vector<float> load_flat_float_values(const std::string& path);

