#!/usr/bin/env python3
"""Apply the opt-in CUDA-graph patch (NAMD_MLFF_CUDA_GRAPH=1) to a copy of
mlff_shim.cpp.  Usage: python3 apply_cuda_graph_patch.py <in.cpp> <out.cpp>"""
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old))
    s = s.replace(old, new)
rep("""#  include <c10/cuda/CUDAFunctions.h>
#  define MLFF_HAVE_CUDA 1""", """#  include <c10/cuda/CUDAFunctions.h>
#  include <c10/cuda/CUDAGuard.h>
#  include <ATen/cuda/CUDAGraph.h>
#  define MLFF_HAVE_CUDA 1""")
rep("""#include <algorithm>
#include <atomic>""", """#include <algorithm>
#include <atomic>
#include <memory>""")
rep("""    /* ---- batch-path buffers ---- */""", """    /* ---- opt-in CUDA-graph path (NAMD_MLFF_CUDA_GRAPH=1, serial non-periodic
     *      calls only; see captureGraph/evalGraph).  The model must export
     *      graph_capacity(coords)->int and graph_step(coords, Z, cap)->(E, F, n_edges)
     *      and carry graph_capable=True (e.g. SchNet exported with --fast). ---- */
    bool    graphCapable  = false;
    int64_t graphMaxAtoms = 0;
    int64_t graphN = -1, graphCap = 0;
    int     graphCaptures = 0;
    torch::Tensor g_coords, g_Z, g_E, g_F, g_n, h_n;
#if MLFF_HAVE_CUDA
    std::unique_ptr<at::cuda::CUDAGraph> graph;
#endif

    /* ---- batch-path buffers ---- */""")
rep("""        ModelImpl *impl = new ModelImpl(std::move(mod), dev, hasBatch);
        impl->takesCell      = (fwdArgs == 6);
        impl->batchTakesCell = (fwdBatchArgs == 8);""", """        ModelImpl *impl = new ModelImpl(std::move(mod), dev, hasBatch);
        impl->takesCell      = (fwdArgs == 6);
        impl->batchTakesCell = (fwdBatchArgs == 8);
#if MLFF_HAVE_CUDA
        /* Opt-in CUDA-graph path: default OFF.  Only for models that export the
         * graph API; anything else keeps the normal forward() path. */
        if (envFlag("NAMD_MLFF_CUDA_GRAPH") == 1 && dev.is_cuda()) {
            auto &m = impl->module;
            bool ok = m.find_method("graph_step").has_value() &&
                      m.find_method("graph_capacity").has_value() &&
                      m.hasattr("graph_capable") && m.attr("graph_capable").toBool();
            if (ok) {
                impl->graphCapable  = true;
                impl->graphMaxAtoms = m.hasattr("graph_max_atoms")
                                          ? m.attr("graph_max_atoms").toInt() : 2048;
                if (const char *mx = std::getenv("NAMD_MLFF_CUDA_GRAPH_MAX_ATOMS"))
                    impl->graphMaxAtoms = std::atoll(mx);
                std::fprintf(stderr, "[mlff_shim] CUDA graph path ON (serial non-periodic "
                             "calls with <= %lld atoms)\\n", (long long)impl->graphMaxAtoms);
            } else {
                std::fprintf(stderr, "[mlff_shim] NAMD_MLFF_CUDA_GRAPH=1 but the model has no "
                             "graph API (graph_step/graph_capacity/graph_capable): using forward()\\n");
            }
        }
#endif""")
rep("""};

}  // namespace

extern "C" {""", """};

#if MLFF_HAVE_CUDA
/* (Re)capture graph_step for the current static inputs.  Warm-up and capture
 * run on a side stream (cuBLAS workspaces are per stream; the profiling
 * executor / NNC fuser must have settled before capture).  Thread-local capture
 * mode, so NAMD's other CUDA threads are not affected. */
static void captureGraph(ModelImpl &M) {
    M.graph.reset();
    M.syncIfCuda();
    M.graphCap = M.module.get_method("graph_capacity")({M.g_coords}).toInt();
    auto side = c10::cuda::getStreamFromPool(false, M.device.index());
    std::vector<torch::jit::IValue> in{M.g_coords, M.g_Z, M.graphCap};
    {
        c10::cuda::CUDAStreamGuard sg(side);
        for (int i = 0; i < 4; ++i) (void)M.module.get_method("graph_step")(in);
    }
    side.synchronize();
    auto g = std::make_unique<at::cuda::CUDAGraph>();
    {
        c10::cuda::CUDAStreamGuard sg(side);
        g->capture_begin({0, 0}, cudaStreamCaptureModeThreadLocal);
        try {
            auto out = M.module.get_method("graph_step")(in).toTuple();
            const auto &els = out->elements();
            M.g_E = els[0].toTensor();
            M.g_F = els[1].toTensor();
            M.g_n = els[2].toTensor();
        } catch (...) {
            try { g->capture_end(); } catch (...) {}
            throw;
        }
        g->capture_end();
    }
    side.synchronize();
    M.graph = std::move(g);
    ++M.graphCaptures;
    if (!M.h_n.defined()) M.h_n = torch::empty({1}, M.hI64());
}

/* One serial step through the captured graph.  Throws on failure (the caller
 * then falls back to forward() for good). */
static void evalGraph(ModelImpl &M, const double *coords, const int64_t *Z, int numQM,
                      double *energy_out, double *forces_out, double *charges_out) {
    if (numQM != M.graphN) {
        M.graph.reset();
        M.g_coords = torch::empty({numQM, 3}, M.dF64());
        M.g_Z      = torch::empty({numQM},    M.dI64());
        M.graphN   = numQM;
    }
    {
        torch::NoGradGuard ng;
        std::memcpy(M.h_coords.data_ptr<double>(), coords, sizeof(double) * numQM * 3);
        std::memcpy(M.h_Z.data_ptr<int64_t>(),     Z,      sizeof(int64_t) * numQM);
        M.g_coords.copy_(M.h_coords.narrow(0, 0, numQM), true);
        M.g_Z.copy_(M.h_Z.narrow(0, 0, numQM), true);
    }
    for (int attempt = 0; attempt < 2; ++attempt) {
        if (!M.graph) captureGraph(M);          /* autograd stays enabled */
        M.graph->replay();
        {
            torch::NoGradGuard ng;
            M.h_energy.copy_(M.g_E.reshape({1}), true);
            M.h_forces.narrow(0, 0, numQM).copy_(M.g_F, true);
            M.h_n.copy_(M.g_n.reshape({1}), true);
        }
        M.syncIfCuda();
        if (M.h_n.data_ptr<int64_t>()[0] <= M.graphCap) break;
        /* More edges than the captured capacity: this result dropped edges.
         * Re-capture with a capacity sized from these coords and redo. */
        M.graph.reset();
        if (attempt == 1) throw std::runtime_error("edge capacity overflow after re-capture");
    }
    energy_out[0] = M.h_energy.data_ptr<double>()[0];
    std::memcpy(forces_out, M.h_forces.data_ptr<double>(), sizeof(double) * numQM * 3);
    if (charges_out) std::memset(charges_out, 0, sizeof(double) * numQM);
}
#endif

}  // namespace

extern "C" {""")
rep("""        M.ensureSerial(numQM, numPC);
        const bool havePC = (numPC > 0 && pc_xyz && pc_q);
""", """        M.ensureSerial(numQM, numPC);
        const bool havePC = (numPC > 0 && pc_xyz && pc_q);
#if MLFF_HAVE_CUDA
        /* Point charges are ignored by every wrapper's forward() as well. */
        if (M.graphCapable && !cell && numQM <= M.graphMaxAtoms) {
            try {
                evalGraph(M, coords, Z, numQM, energy_out, forces_out, charges_out);
                return 0;
            } catch (const std::exception &e) {
                std::fprintf(stderr, "[mlff_shim] CUDA graph path failed (%s); "
                             "falling back to forward() for this model\\n", e.what());
                M.graphCapable = false;
                M.graph.reset();
                M.syncIfCuda();
            }
        }
#endif
""")
open(dst, "w").write(s)
print("patched ->", dst)
