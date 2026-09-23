// Interleaved PJRT benchmark for FeNNiX StableHLO artifacts -- the same C++ runtime NAMD uses
// (namd_fennix/src/fennix_pjrt), same client options (preallocate=0, memory_fraction), no Python.
//
// usage: bench_pjrt [--plugin P] [--mem-fraction F] [--iters N] [--warmup W] [--jitter A]
//                   [--exec namd|fast] [--out result.json] label=manifest.json [label2=...]
//
//  * every model gets the SAME jittered geometries (reference coords + uniform(-A,A), fixed seed),
//    models are interleaved per iteration (laptop clocks drift), time = one full evaluate() as in
//    ComputeFennix.C (upload + execute + download + f32->f64 kcal conversion).
//  * --exec namd : PjrtPlugin::execute_compiled exactly as NAMD calls it (baseline backend path)
//    --exec fast : the patched evaluate path (scripts/opt/fennix/namd_patch): one upload with
//                  kImmutableOnlyDuringCall, no separate execute await, both outputs copied
//                  straight into preallocated host arrays (no size query), awaited together.
//  * parity: max|dE| (kcal/mol) and max|dF| (kcal/mol/A) of each model vs the FIRST model over all
//    geometries; peak device memory via PJRT_Device_MemoryStats (per model: cleared before its
//    first call is not possible across models, so we report bytes_in_use peak for the client).
#include "pjrt_plugin.h"
#include "artifact_bundle.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template <typename T, size_t N>
T mk() { T a; std::memset(&a, 0, sizeof(a)); a.struct_size = N; return a; }
#define MK(type) mk<type, type##_STRUCT_SIZE>()

struct Model {
    std::string label;
    ArtifactSpec spec;
    PjrtExecutableHandle exe;
    std::vector<double> times_ms;
    std::vector<double> energies;          // kcal/mol per geometry
    std::vector<std::vector<double>> forces;
    double compile_s = 0;
    int64_t peak_bytes = -1;
    std::string exec;                      // "namd" | "fast" (per model; default --exec)
};

void chk(const PJRT_Api* api, PJRT_Error* e, const char* what) {
    if (!e) return;
    auto m = MK(PJRT_Error_Message_Args); m.error = e; api->PJRT_Error_Message(&m);
    std::string s(m.message, m.message_size);
    auto d = MK(PJRT_Error_Destroy_Args); d.error = e; api->PJRT_Error_Destroy(&d);
    throw std::runtime_error(std::string(what) + ": " + s);
}

void await_destroy(const PJRT_Api* api, PJRT_Event* ev) {
    if (!ev) return;
    auto a = MK(PJRT_Event_Await_Args); a.event = ev; chk(api, api->PJRT_Event_Await(&a), "await");
    auto d = MK(PJRT_Event_Destroy_Args); d.event = ev; chk(api, api->PJRT_Event_Destroy(&d), "evdestroy");
}

void destroy_buf(const PJRT_Api* api, PJRT_Buffer* b) {
    if (!b) return;
    auto d = MK(PJRT_Buffer_Destroy_Args); d.buffer = b; chk(api, api->PJRT_Buffer_Destroy(&d), "bufdestroy");
}

// The patched NAMD evaluate path (see namd_patch/). outputs: energy (1 float), forces (n*3 floats).
void execute_fast(const PjrtPlugin& pl, const PjrtExecutableHandle& exe, const std::vector<float>& in,
                  const std::vector<int64_t>& dims, std::vector<float>& e_out, std::vector<float>& f_out,
                  PJRT_Device* dev) {
    const PJRT_Api* api = pl.api();
    auto up = MK(PJRT_Client_BufferFromHostBuffer_Args);
    up.client = pl.client();
    up.data = in.data();
    up.type = PJRT_Buffer_Type_F32;
    up.dims = dims.data();
    up.num_dims = dims.size();
    up.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
    up.device = dev;
    chk(api, api->PJRT_Client_BufferFromHostBuffer(&up), "upload");
    PJRT_Buffer* inb = up.buffer;
    await_destroy(api, up.done_with_host_buffer);  // already complete for OnlyDuringCall

    auto opts = MK(PJRT_ExecuteOptions);
    opts.launch_id = 0;
    opts.call_location = "fennix_fast";
    opts.use_major_to_minor_data_layout_for_callbacks = true;
    PJRT_Buffer* args1[1] = {inb};
    PJRT_Buffer* const* argl[1] = {args1};
    std::vector<PJRT_Buffer*> outs(exe.num_outputs, nullptr);
    PJRT_Buffer** outl[1] = {outs.data()};
    auto ex = MK(PJRT_LoadedExecutable_Execute_Args);
    ex.executable = exe.loaded;
    ex.options = &opts;
    ex.argument_lists = argl;
    ex.num_devices = 1;
    ex.num_args = 1;
    ex.output_lists = outl;
    ex.device_complete_events = nullptr;  // no completion event: the D2H copies order after it
    ex.execute_device = dev;
    chk(api, api->PJRT_LoadedExecutable_Execute(&ex), "execute");

    std::vector<float>* dst[2] = {&e_out, &f_out};
    PJRT_Event* evs[2] = {nullptr, nullptr};
    for (int k = 0; k < 2; ++k) {
        auto c = MK(PJRT_Buffer_ToHostBuffer_Args);
        c.src = outs[k];
        c.dst = dst[k]->data();
        c.dst_size = dst[k]->size() * sizeof(float);
        chk(api, api->PJRT_Buffer_ToHostBuffer(&c), "d2h");
        evs[k] = c.event;
    }
    for (int k = 0; k < 2; ++k) await_destroy(api, evs[k]);
    for (PJRT_Buffer* b : outs) destroy_buf(api, b);
    destroy_buf(api, inb);
}

PJRT_Device* first_dev(const PjrtPlugin& pl) { return pl.first_addressable_device(); }

int64_t peak_mem(const PjrtPlugin& pl, PJRT_Device* dev) {
    const PJRT_Api* api = pl.api();
    if (api->struct_size <= offsetof(PJRT_Api, PJRT_Device_MemoryStats) || !api->PJRT_Device_MemoryStats) return -1;
    auto a = MK(PJRT_Device_MemoryStats_Args);
    a.device = dev;
    PJRT_Error* e = api->PJRT_Device_MemoryStats(&a);
    if (e) { auto d = MK(PJRT_Error_Destroy_Args); d.error = e; api->PJRT_Error_Destroy(&d); return -1; }
    return a.peak_bytes_in_use_is_set ? a.peak_bytes_in_use : a.bytes_in_use;
}

// Per-executable device footprint from the compiler (deterministic, per model):
// generated code + argument + output + temp - alias.  Uses the older, smaller Args prefix
// (fields up to host_temp_size_in_bytes) so it also works with older plugins.
int64_t compiled_mem(const PjrtPlugin& pl, const PjrtExecutableHandle& exe, int64_t* temp_out) {
    const PJRT_Api* api = pl.api();
    if (api->struct_size <= offsetof(PJRT_Api, PJRT_Executable_GetCompiledMemoryStats) ||
        !api->PJRT_Executable_GetCompiledMemoryStats) return -1;
    PJRT_Executable_GetCompiledMemoryStats_Args a;
    std::memset(&a, 0, sizeof(a));
    a.struct_size = offsetof(PJRT_Executable_GetCompiledMemoryStats_Args, host_temp_size_in_bytes) + sizeof(int64_t);
    a.executable = exe.executable;
    PJRT_Error* e = api->PJRT_Executable_GetCompiledMemoryStats(&a);
    if (e) { auto d = MK(PJRT_Error_Destroy_Args); d.error = e; api->PJRT_Error_Destroy(&d); return -1; }
    if (temp_out) *temp_out = a.temp_size_in_bytes;
    return a.generated_code_size_in_bytes + a.argument_size_in_bytes + a.output_size_in_bytes +
           a.temp_size_in_bytes - a.alias_size_in_bytes;
}

double pct(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    if (v.empty()) return NAN;
    double idx = p * (v.size() - 1);
    size_t lo = (size_t)std::floor(idx), hi = (size_t)std::ceil(idx);
    return v[lo] + (v[hi] - v[lo]) * (idx - lo);
}

}  // namespace

int main(int argc, char** argv) {
    std::string plugin = std::getenv("NAMD_FENNIX_PLUGIN") ? std::getenv("NAMD_FENNIX_PLUGIN") : "";
    double memfrac = 0.15;
    int iters = 50, warmup = 5;
    double jitter = 0.02;
    std::string exec_mode = "namd", out;
    std::vector<std::pair<std::string, std::string>> specs;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) throw std::runtime_error("missing value for " + a); return std::string(argv[++i]); };
        if (a == "--plugin") plugin = next();
        else if (a == "--mem-fraction") memfrac = std::atof(next().c_str());
        else if (a == "--iters") iters = std::atoi(next().c_str());
        else if (a == "--warmup") warmup = std::atoi(next().c_str());
        else if (a == "--jitter") jitter = std::atof(next().c_str());
        else if (a == "--exec") exec_mode = next();
        else if (a == "--out") out = next();
        else {
            auto p = a.find('=');
            if (p == std::string::npos) throw std::runtime_error("bad arg " + a);
            specs.push_back({a.substr(0, p), a.substr(p + 1)});  // manifest may end in @namd / @fast
        }
    }
    if (specs.empty() || plugin.empty()) { std::fprintf(stderr, "need --plugin and label=manifest\n"); return 2; }

    PjrtPlugin pl(plugin);
    pl.load();
    pl.initialize();
    PjrtClientOptions opts;
    opts.preallocate = 0;
    opts.memory_fraction = memfrac;
    if (const char* al = std::getenv("FENNIX_ALLOC")) if (*al) opts.allocator = al;
    pl.create_client(opts);
    PJRT_Device* dev = first_dev(pl);

    std::vector<Model> models(specs.size());
    for (size_t m = 0; m < specs.size(); ++m) {
        models[m].label = specs[m].first;
        std::string path = specs[m].second;
        models[m].exec = exec_mode;
        auto at = path.rfind('@');
        if (at != std::string::npos) { models[m].exec = path.substr(at + 1); path = path.substr(0, at); }
        models[m].spec = load_artifact_spec(path);
        auto t0 = std::chrono::steady_clock::now();
        models[m].exe = pl.compile_mlir(read_text_file(models[m].spec.stablehlo_path),
                                        read_text_file(models[m].spec.compile_options_path));
        models[m].compile_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "[bench] compiled %s in %.1f s\n", models[m].label.c_str(), models[m].compile_s);
    }
    // per-model reference geometry (batched [W,N,3] artifacts carry their own); walker 0 of the
    // exporter's default walker set == the single-walker reference, so parity of a [W,N,3] model vs a
    // [N,3] model 0 is taken on walker 0 (identical jittered geometry: same jitter vector prefix).
    std::vector<RuntimeReference> refs;
    size_t maxin = 0;
    for (Model& M : models) {
        refs.push_back(load_runtime_reference(M.spec.reference_runtime_path));
        maxin = std::max(maxin, refs.back().coordinates.size());
    }
    const RuntimeReference& ref = refs[0];
    const size_t nin = ref.coordinates.size();
    const size_t nf = nin;  // forces same shape as coords
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> U(-(float)jitter, (float)jitter);

    std::vector<float> e_host(1), f_host(nf);
    if (std::getenv("BENCH_CHECK_EXEC")) {
        // same executable + same input through both host paths: max diff vs run-to-run floor
        double floorE = 0, floorF = 0, dE = 0, dF = 0;
        for (int g = 0; g < 20; ++g) {
            std::vector<float> x = ref.coordinates;
            for (float& v : x) v += U(rng);
            Model& M = models[0];
            PjrtExecutionResult a = pl.execute_compiled(M.exe, x, M.spec.input_dims);
            PjrtExecutionResult b = pl.execute_compiled(M.exe, x, M.spec.input_dims);
            execute_fast(pl, M.exe, x, M.spec.input_dims, e_host, f_host, dev);
            floorE = std::max(floorE, (double)std::fabs(a.outputs[0].f32_values[0] - b.outputs[0].f32_values[0]));
            dE = std::max(dE, (double)std::fabs(a.outputs[0].f32_values[0] - e_host[0]));
            for (size_t i = 0; i < nf; ++i) {
                floorF = std::max(floorF, (double)std::fabs(a.outputs[1].f32_values[i] - b.outputs[1].f32_values[i]));
                dF = std::max(dF, (double)std::fabs(a.outputs[1].f32_values[i] - f_host[i]));
            }
        }
        std::printf("CHECK_EXEC (eV, eV/A) namd-vs-namd floor: dE %.3e dF %.3e | namd-vs-fast: dE %.3e dF %.3e\n",
                    floorE, floorF, dE, dF);
    }
    for (int it = -warmup; it < iters; ++it) {
        std::vector<float> jit(maxin);
        for (float& v : jit) v = U(rng);
        for (size_t m = 0; m < models.size(); ++m) {
            Model& M = models[m];
            std::vector<float> x = refs[m].coordinates;
            for (size_t i = 0; i < x.size(); ++i) x[i] += jit[i];
            const size_t mf = x.size();
            size_t me = 1;
            for (int64_t d : M.spec.energy_dims) me *= (d > 0 ? (size_t)d : 1);
            std::vector<float> eb(me), fb(mf);
            auto t0 = std::chrono::steady_clock::now();
            std::vector<double> ek(me), fk(mf);
            if (M.exec == "namd") {
                PjrtExecutionResult r = pl.execute_compiled(M.exe, x, M.spec.input_dims);
                for (size_t i = 0; i < me; ++i) ek[i] = (double)r.outputs[0].f32_values[i] * M.spec.ev_to_kcal;
                for (size_t i = 0; i < mf; ++i) fk[i] = (double)r.outputs[1].f32_values[i] * M.spec.ev_to_kcal;
            } else {
                execute_fast(pl, M.exe, x, M.spec.input_dims, eb, fb, dev);
                for (size_t i = 0; i < me; ++i) ek[i] = (double)eb[i] * M.spec.ev_to_kcal;
                for (size_t i = 0; i < mf; ++i) fk[i] = (double)fb[i] * M.spec.ev_to_kcal;
            }
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (it >= 0) {
                M.times_ms.push_back(ms);
                M.energies.push_back(ek[0]);   // walker 0
                fk.resize(nf <= mf ? nf : mf); // walker 0 slice (compared vs model 0)
                M.forces.push_back(std::move(fk));
            }
        }
    }
    int64_t pk = peak_mem(pl, dev);

    std::ostringstream js;
    js << "{\n  \"n_atoms_model0_input\": " << nin / 3 << ", \"iters\": " << iters << ", \"jitter\": " << jitter
       << ", \"exec\": \"" << exec_mode << "\", \"mem_fraction\": " << memfrac
       << ", \"client_peak_bytes\": " << pk << ",\n  \"models\": [\n";
    std::printf("%-22s %9s %9s %9s %9s %12s %12s\n", "model", "median_ms", "p10", "p90", "compile_s", "max|dE|", "max|dF|");
    for (size_t m = 0; m < models.size(); ++m) {
        Model& M = models[m];
        double dE = 0, dF = 0;
        for (size_t g = 0; g < M.energies.size(); ++g) {
            dE = std::max(dE, std::fabs(M.energies[g] - models[0].energies[g]));
            const size_t nc = std::min(M.forces[g].size(), models[0].forces[g].size());
            for (size_t i = 0; i < nc; ++i) dF = std::max(dF, std::fabs(M.forces[g][i] - models[0].forces[g][i]));
        }
        double med = pct(M.times_ms, 0.5), p10 = pct(M.times_ms, 0.1), p90 = pct(M.times_ms, 0.9);
        int64_t tmp = -1, foot = compiled_mem(pl, M.exe, &tmp);
        std::printf("%-22s %9.3f %9.3f %9.3f %9.1f %12.3e %12.3e   E0=%.4f  dev_MiB=%.1f (temp %.1f)\n", M.label.c_str(), med, p10, p90,
                    M.compile_s, dE, dF, M.energies[0], foot / 1048576.0, tmp / 1048576.0);
        js << "    {\"label\": \"" << M.label << "\", \"manifest\": \"" << M.spec.manifest_path
           << "\", \"exec\": \"" << M.exec << "\", \"median_ms\": " << med << ", \"p10_ms\": " << p10 << ", \"p90_ms\": " << p90
           << ", \"compile_s\": " << M.compile_s << ", \"max_dE_kcal\": " << dE << ", \"max_dF_kcal_A\": " << dF
           << ", \"E0_kcal\": " << M.energies[0] << ", \"compiled_device_bytes\": " << foot << ", \"temp_bytes\": " << tmp << "}" << (m + 1 < models.size() ? "," : "") << "\n";
    }
    js << "  ]\n}\n";
    std::printf("client peak device bytes: %lld (%.1f MiB)\n", (long long)pk, pk / 1048576.0);
    if (!out.empty()) { std::ofstream(out) << js.str(); std::fprintf(stderr, "[bench] wrote %s\n", out.c_str()); }
    std::fflush(stdout);
    std::_Exit(0);  // skip PJRT teardown (crash-prone at exit, same as NAMD's leak-by-design)
}
