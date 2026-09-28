// libmace_cueq_pyinit.so -- makes cuEquivariance TorchScript artifacts loadable
// in a process without Python (NAMD's libtorch shim).
//
// cuequivariance::uniform_1d is registered by a Python torch.library.custom_op
// (cuequivariance_ops_torch/uniform_1d.py), so the op only exists once
// `import cuequivariance_torch` has run.  This library does that from a static
// constructor when it is dlopen'd (NAMD_MLFF_EXTRA_LIBS, LAST entry), before the
// shim calls torch::jit::load.  Needs the pip-torch shim (libnamd_mlff_pytorch.so):
// importing torch in Python must see the same libtorch the shim uses.
//
// In a process that already runs Python (bench_common --extra-lib) it just
// imports the module under the GIL.  Python is never finalised.
//
// Env: MACE_CUEQ_PYHOME  (default /home/rat/miniconda3/envs/allegro) = sys.prefix
#include <Python.h>
#include <cstdio>
#include <cstdlib>

namespace {
struct PyInit {
  PyInit() {
    bool own = false;
    if (!Py_IsInitialized()) {
      const char* home = std::getenv("MACE_CUEQ_PYHOME");
      if (!home || !*home) home = "/home/rat/miniconda3/envs/allegro";
      PyConfig config;
      PyConfig_InitPythonConfig(&config);
      config.install_signal_handlers = 0;   // leave NAMD's / charm's handlers alone
      config.parse_argv = 0;
      PyStatus st = PyConfig_SetBytesString(&config, &config.home, home);
      if (!PyStatus_Exception(st)) st = Py_InitializeFromConfig(&config);
      PyConfig_Clear(&config);
      if (PyStatus_Exception(st)) {
        std::fprintf(stderr, "[mace_cueq_pyinit] Py_InitializeFromConfig failed: %s\n",
                     st.err_msg ? st.err_msg : "?");
        std::abort();
      }
      own = true;
    }
    PyGILState_STATE g = PyGILState_Ensure();
    PyObject* m = PyImport_ImportModule("cuequivariance_torch");
    if (!m) {
      std::fprintf(stderr, "[mace_cueq_pyinit] import cuequivariance_torch failed:\n");
      PyErr_Print();
      std::abort();
    }
    Py_DECREF(m);
    PyGILState_Release(g);
    // We own the interpreter: release the GIL held by this (main) thread so
    // the torch Python-op kernel can take it from whatever thread NAMD uses.
    if (own) PyEval_SaveThread();
    std::fprintf(stderr, "[mace_cueq_pyinit] cuequivariance_torch imported (%s interpreter)\n",
                 own ? "embedded" : "host");
  }
} g_init;
}  // namespace
