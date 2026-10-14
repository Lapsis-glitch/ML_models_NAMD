// LD_PRELOAD interposer: logs every run_uniform_1d_cuda call made by the
// cuequivariance_ops_torch pybind layer (to learn its flat->nested mapping).
#include <dlfcn.h>
#include <cstdio>
#include <string>
#include <vector>
#include "cuequivariance_ops/equivariance/uniform1d/api.hh"
namespace U = kernelcatcher::equivariance::uniform_1d;
using kernelcatcher::utils::Datatype;
using kernelcatcher::utils::BatchDimension;
template <class T> static void pv(const char* n, const std::vector<T>& v) {
  std::fprintf(stderr, " %s=[", n);
  for (auto& x : v) std::fprintf(stderr, "%g,", (double)x);
  std::fprintf(stderr, "]");
}
template <class T> static void pvv(const char* n, const std::vector<std::vector<T>>& v) {
  std::fprintf(stderr, " %s=[", n);
  for (auto& x : v) { std::fprintf(stderr, "["); for (auto& y : x) std::fprintf(stderr, "%d,", (int)y); std::fprintf(stderr, "],"); }
  std::fprintf(stderr, "]");
}
namespace kernelcatcher::equivariance::uniform_1d {
int run_uniform_1d_cuda(KC_UNIFORM_1D_CPU_ARGUMENTS, bool ignore_first, void* stream) {
  std::fprintf(stderr, "SPY name=%s math=%d ext=%d nin=%d nout=%d nidx=%d", name.c_str(), (int)math_dtype,
               operand_extent, num_inputs, num_outputs, num_index);
  std::vector<int> bd; for (auto d : buffer_dim) bd.push_back((int)d); pv("buffer_dim", bd);
  pv("nseg", buffer_num_segments); pvv("batch_dim", batch_dim); pvv("index_cfg", index_configuration);
  pv("index_extent", index_extent); std::vector<int> dts; for (auto d : dtypes) dts.push_back((int)d); pv("dtypes", dts);
  pvv("ops", operations); pv("num_paths", num_paths); pv("pis", path_indices_start); pv("pcs", path_coefficients_start);
  std::fprintf(stderr, " n_path_idx=%zu n_path_coef=%zu", path_indices.size(), path_coefficients.size());
  pv("batch_sizes", batch_sizes); pv("buffer_bytes", buffer_bytes);
  std::fprintf(stderr, " nbuf=%zu zero_out=%d ignore_first=%d\n", buffers.size(), (int)zero_output_buffers, (int)ignore_first);
  using F = int (*)(KC_UNIFORM_1D_CPU_ARGUMENTS, bool, void*);
  static void* h = dlopen("libcue_ops.so", RTLD_NOW | RTLD_NOLOAD);
  static F real = (F)dlvsym(h, "_ZN13kernelcatcher12equivariance10uniform_1d19run_uniform_1d_cudaERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEENS_5utils8DatatypeEiiiiRKSt6vectorINS1_9DimensionESaISD_EERKSC_IiSaIiEERKSC_ISC_INSA_14BatchDimensionESaISM_EESaISO_EERKSC_ISJ_SaISJ_EESL_RKSC_ISB_SaISB_EESW_SL_SL_SL_SL_RKSC_IdSaIdEESL_RKSC_IPvSaIS15_EERKSC_ImSaImEEbbS15_", "all");
  return real(name, math_dtype, operand_extent, num_inputs, num_outputs, num_index, buffer_dim, buffer_num_segments,
              batch_dim, index_configuration, index_extent, dtypes, operations, num_paths, path_indices_start,
              path_coefficients_start, path_indices, path_coefficients, batch_sizes, buffers, buffer_bytes,
              zero_output_buffers, ignore_first, stream);
}
}
