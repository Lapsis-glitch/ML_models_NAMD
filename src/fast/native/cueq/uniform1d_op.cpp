// Native (Python-free) registration of the cuEquivariance op
//   cuequivariance::uniform_1d
// so that cuEq-accelerated TorchScript artifacts load in a pure-libtorch process
// (the NAMD shim) with  NAMD_MLFF_EXTRA_LIBS=libcue_ops.so:libcueq_uniform1d_native.so
//
// Upstream the op is a Python torch.library.custom_op
// (cuequivariance_ops_torch/uniform_1d.py) whose CUDA work is a single call to
// kernelcatcher::equivariance::uniform_1d::run_uniform_1d_cuda in libcue_ops.so.
// This file is a line-by-line C++ port of that Python op:
//   * forward  = the custom_op body (+ _handle_batch_dim_auto, output allocation,
//                flat->nested argument mapping of the pybind layer, recorded with
//                spy.cpp in spy_mapping.log)
//   * backward = _do_bwd_jit (the backward is the same op with a "_bwd" name and
//                rewired operations), dispatched through the op again so double
//                backward also works.
// Do NOT load this library in a process that also imports cuequivariance_ops_torch
// (both would register the same op name).
#include <torch/library.h>
#include <torch/autograd.h>
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <string>
#include <vector>

#include "cuequivariance_ops/equivariance/uniform1d/api.hh"

namespace U = kernelcatcher::equivariance::uniform_1d;
using kernelcatcher::utils::BatchDimension;
using kernelcatcher::utils::Datatype;

namespace {

constexpr int64_t BD_SHARED = 0, BD_BATCHED = 1, BD_INDEXED = -1, BD_AUTO = -2;
constexpr int kNumNonTensorArgs = 20;  // everything before `tensors` in the schema

using IntVec = std::vector<int64_t>;

IntVec vec(c10::IntArrayRef a) { return IntVec(a.begin(), a.end()); }

Datatype map_dtype(at::ScalarType t) {
  switch (t) {
    case at::kDouble: return Datatype::kFloat64;
    case at::kFloat: return Datatype::kFloat32;
    case at::kHalf: return Datatype::kFloat16;
    case at::kBFloat16: return Datatype::kBFloat16;
    case at::kInt: return Datatype::kInt32;
    case at::kLong: return Datatype::kInt64;
    default: TORCH_CHECK(false, "cuequivariance::uniform_1d (native): unsupported dtype ", t);
  }
}

BatchDimension map_batch_dim(int64_t b) {
  if (b == BD_BATCHED) return BatchDimension::kBatched;
  if (b == BD_SHARED) return BatchDimension::kShared;
  if (b == BD_INDEXED) return BatchDimension::kIndexed;
  TORCH_CHECK(false, "cuequivariance::uniform_1d (native): unknown batch dim ", b);
}

U::Dimension map_buffer_dim(int64_t o) {
  if (o == 0) return U::Dimension::kScalar;
  if (o == 1) return U::Dimension::kOneDimensional;
  TORCH_CHECK(false, "cuequivariance::uniform_1d (native): unknown buffer dim ", o);
}

// port of _handle_batch_dim_auto
std::pair<int64_t, IntVec> handle_batch_dim_auto(int64_t batch_size, IntVec batch_dim,
                                                 at::TensorList inputs, at::TensorList index_tensors) {
  int64_t bs = batch_size;
  for (const auto& t : index_tensors) {
    if (bs == BD_AUTO) bs = t.size(0);
    else TORCH_CHECK(bs == t.size(0), "batch dim mismatch");
  }
  IntVec nbd = batch_dim;
  for (size_t idx = 0; idx < inputs.size(); ++idx) {
    const int64_t bd = batch_dim[idx];
    const auto& t = inputs[idx];
    if (bd == BD_SHARED) {
      TORCH_CHECK(t.size(0) == 1, "shared batch dim must be 1");
    } else if (bd == BD_INDEXED) {
      continue;
    } else if (bd == BD_BATCHED) {
      if (bs == BD_AUTO) bs = t.size(0);
      else TORCH_CHECK(bs == t.size(0), "batch dim mismatch");
    } else if (bd == BD_AUTO) {
      if (t.size(0) == 1) {
        nbd[idx] = BD_SHARED;
      } else {
        if (bs == BD_AUTO) bs = t.size(0);
        else TORCH_CHECK(bs == t.size(0), "batch dim mismatch");
        nbd[idx] = BD_BATCHED;
      }
    } else {
      TORCH_CHECK(false, "Unknown batch dim kind ", bd);
    }
  }
  for (size_t idx = inputs.size(); idx < batch_dim.size(); ++idx)
    if (batch_dim[idx] == BD_AUTO) nbd[idx] = BD_BATCHED;
  if (bs == BD_AUTO) bs = 1;
  if (bs == 1)
    for (auto& b : nbd)
      if (b == BD_SHARED) b = BD_BATCHED;
  return {bs, nbd};
}

// port of the custom_op body (CUDA)
std::vector<at::Tensor> uniform_1d_cuda(
    c10::string_view name_sv, at::ScalarType math_dtype, int64_t operand_extent, int64_t num_inputs,
    int64_t num_outputs, int64_t num_index, c10::IntArrayRef buffer_dim, c10::IntArrayRef buffer_num_segments,
    c10::IntArrayRef batch_dim_in, c10::IntArrayRef index_buffer, c10::IntArrayRef dtypes, int64_t num_operations,
    c10::IntArrayRef num_operands, c10::IntArrayRef operations, c10::IntArrayRef num_paths,
    c10::IntArrayRef path_indices_start, c10::IntArrayRef path_coefficients_start, c10::IntArrayRef path_indices,
    c10::ArrayRef<double> path_coefficients, int64_t batch_size_in, at::TensorList tensors_in) {
  TORCH_CHECK(!tensors_in.empty() && tensors_in[0].is_cuda(), "uniform_1d (native): CUDA tensors required");
  const int64_t nin = num_inputs, nout = num_outputs, nio = nin + nout;
  auto [batch_size, batch_dim] = handle_batch_dim_auto(batch_size_in, vec(batch_dim_in),
                                                       tensors_in.slice(0, nin), tensors_in.slice(nin));
  const c10::cuda::CUDAGuard guard(tensors_in[0].device());
  const auto dev = tensors_in[0].device();

  std::vector<at::Tensor> outputs;
  outputs.reserve(nout);
  for (int64_t i = nin; i < nio; ++i) {
    int64_t size0 = 0, size1 = 0;
    if (batch_dim[i] == BD_SHARED) size0 = 1;
    else if (batch_dim[i] == BD_BATCHED) size0 = batch_size;
    else if (batch_dim[i] == BD_INDEXED) size0 = index_buffer[index_buffer[i] + nio];
    if (buffer_dim[i] == 0) size1 = buffer_num_segments[i];
    if (buffer_dim[i] == 1) size1 = operand_extent * buffer_num_segments[i];
    const at::ScalarType dt = dtypes[i] == -1 ? math_dtype : tensors_in[dtypes[i]].scalar_type();
    auto o = at::empty({size0, size1}, at::TensorOptions().dtype(dt).device(dev));
    if (batch_dim[i] == BD_SHARED || batch_dim[i] == BD_INDEXED) o.zero_();
    outputs.push_back(std::move(o));
  }

  // buffers = inputs + outputs + index tensors  (all contiguous)
  std::vector<at::Tensor> bufs;
  bufs.reserve(tensors_in.size() + nout);
  for (int64_t i = 0; i < nin; ++i) bufs.push_back(tensors_in[i].contiguous());
  for (auto& o : outputs) bufs.push_back(o);
  for (size_t i = nin; i < tensors_in.size(); ++i) bufs.push_back(tensors_in[i].contiguous());
  const size_t nbuf = bufs.size();

  // flat -> nested mapping of the pybind layer (see spy_mapping.log)
  std::vector<U::Dimension> bdim;
  for (auto b : buffer_dim) bdim.push_back(map_buffer_dim(b));
  std::vector<int> nseg(buffer_num_segments.begin(), buffer_num_segments.end());
  std::vector<std::vector<BatchDimension>> bdn;
  std::vector<std::vector<int>> icfg;
  for (size_t i = 0; i < nbuf; ++i) {
    if ((int64_t)i < nio) {
      bdn.push_back({map_batch_dim(batch_dim[i])});
      icfg.push_back({(int)index_buffer[i]});
    } else {  // index buffers
      bdn.push_back({BatchDimension::kBatched});
      icfg.push_back({-1});
    }
  }
  std::vector<int> iext(index_buffer.begin() + nio, index_buffer.end());
  std::vector<Datatype> dts;
  for (auto& b : bufs) dts.push_back(map_dtype(b.scalar_type()));
  std::vector<std::vector<int>> ops;
  {
    size_t k = 0;
    for (int64_t i = 0; i < num_operations; ++i) {
      ops.emplace_back(operations.begin() + k, operations.begin() + k + num_operands[i]);
      k += num_operands[i];
    }
  }
  std::vector<int> npaths(num_paths.begin(), num_paths.end());
  std::vector<int> pis(path_indices_start.begin(), path_indices_start.end());
  std::vector<int> pcs(path_coefficients_start.begin(), path_coefficients_start.end());
  std::vector<int> pidx(path_indices.begin(), path_indices.end());
  std::vector<double> pcoef(path_coefficients.begin(), path_coefficients.end());
  std::vector<int> bsizes{(int)batch_size};
  std::vector<void*> ptrs;
  std::vector<size_t> bytes;
  for (auto& b : bufs) {
    ptrs.push_back(b.data_ptr());
    bytes.push_back(b.nbytes());
  }
  (void)num_index;
  const std::string name(name_sv);
  auto stream = c10::cuda::getCurrentCUDAStream(dev.index()).stream();
  const int rc = U::run_uniform_1d_cuda(name, map_dtype(math_dtype), (int)operand_extent, (int)nin, (int)nout,
                                        (int)num_index, bdim, nseg, bdn, icfg, iext, dts, ops, npaths, pis, pcs,
                                        pidx, pcoef, bsizes, ptrs, bytes, /*zero_output_buffers=*/false,
                                        /*ignore_first_batch_size_in_cache_key=*/true, stream);
  TORCH_CHECK(rc == 0, "cuequivariance::uniform_1d (native): run_uniform_1d_cuda returned ", rc, " for ", name);
  return outputs;
}

std::vector<at::Tensor> call_op(const std::string& name, at::ScalarType math_dtype, int64_t operand_extent,
                                int64_t nin, int64_t nout, int64_t nidx, const IntVec& buffer_dim,
                                const IntVec& nseg, const IntVec& batch_dim, const IntVec& index_buffer,
                                const IntVec& dtypes, int64_t nops, const IntVec& num_operands,
                                const IntVec& operations, const IntVec& num_paths, const IntVec& pis,
                                const IntVec& pcs, const IntVec& path_indices, const std::vector<double>& pcoef,
                                int64_t batch_size, const std::vector<at::Tensor>& tensors) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("cuequivariance::uniform_1d", "")
                       .typed<std::vector<at::Tensor>(
                           c10::string_view, at::ScalarType, int64_t, int64_t, int64_t, int64_t, c10::IntArrayRef,
                           c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef, int64_t,
                           c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef,
                           c10::IntArrayRef, c10::ArrayRef<double>, int64_t, at::TensorList)>();
  return op.call(name, math_dtype, operand_extent, nin, nout, nidx, buffer_dim, nseg, batch_dim, index_buffer,
                 dtypes, nops, num_operands, operations, num_paths, pis, pcs, path_indices, pcoef, batch_size,
                 tensors);
}

class Uniform1dFn : public torch::autograd::Function<Uniform1dFn> {
 public:
  static std::vector<at::Tensor> forward(
      torch::autograd::AutogradContext* ctx, c10::string_view name, at::ScalarType math_dtype,
      int64_t operand_extent, int64_t num_inputs, int64_t num_outputs, int64_t num_index,
      c10::IntArrayRef buffer_dim, c10::IntArrayRef buffer_num_segments, c10::IntArrayRef batch_dim,
      c10::IntArrayRef index_buffer, c10::IntArrayRef dtypes, int64_t num_operations, c10::IntArrayRef num_operands,
      c10::IntArrayRef operations, c10::IntArrayRef num_paths, c10::IntArrayRef path_indices_start,
      c10::IntArrayRef path_coefficients_start, c10::IntArrayRef path_indices,
      c10::ArrayRef<double> path_coefficients, int64_t batch_size, at::TensorList tensors) {
    auto& sd = ctx->saved_data;
    sd["name"] = std::string(name);
    sd["math_dtype"] = (int64_t)math_dtype;
    sd["operand_extent"] = operand_extent;
    sd["num_inputs"] = num_inputs;
    sd["num_outputs"] = num_outputs;
    sd["num_index"] = num_index;
    sd["buffer_dim"] = vec(buffer_dim);
    sd["nseg"] = vec(buffer_num_segments);
    sd["batch_dim"] = vec(batch_dim);
    sd["index_buffer"] = vec(index_buffer);
    sd["dtypes"] = vec(dtypes);
    sd["num_operations"] = num_operations;
    sd["num_operands"] = vec(num_operands);
    sd["operations"] = vec(operations);
    sd["num_paths"] = vec(num_paths);
    sd["pis"] = vec(path_indices_start);
    sd["pcs"] = vec(path_coefficients_start);
    sd["path_indices"] = vec(path_indices);
    sd["path_coefficients"] = std::vector<double>(path_coefficients.begin(), path_coefficients.end());
    sd["batch_size"] = batch_size;
    ctx->save_for_backward(tensors.vec());
    auto outs = uniform_1d_cuda(name, math_dtype, operand_extent, num_inputs, num_outputs, num_index, buffer_dim,
                                buffer_num_segments, batch_dim, index_buffer, dtypes, num_operations, num_operands,
                                operations, num_paths, path_indices_start, path_coefficients_start, path_indices,
                                path_coefficients, batch_size, tensors);
    return outs;
  }

  // port of _do_bwd_jit
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                 torch::autograd::variable_list grad) {
    auto& sd = ctx->saved_data;
    const std::string oname = sd["name"].toStringRef();
    const auto math_dtype = (at::ScalarType)sd["math_dtype"].toInt();
    const int64_t ext = sd["operand_extent"].toInt();
    const int64_t nin = sd["num_inputs"].toInt(), nout = sd["num_outputs"].toInt();
    const int64_t nidx = sd["num_index"].toInt();
    const IntVec obdim = sd["buffer_dim"].toIntVector();
    const IntVec onseg = sd["nseg"].toIntVector();
    IntVec obatch = sd["batch_dim"].toIntVector();
    const IntVec oindex = sd["index_buffer"].toIntVector();
    const IntVec odtypes = sd["dtypes"].toIntVector();
    const int64_t onops = sd["num_operations"].toInt();
    const IntVec onoperands = sd["num_operands"].toIntVector();
    const IntVec ooperations = sd["operations"].toIntVector();
    const IntVec onpaths = sd["num_paths"].toIntVector();
    const IntVec opis = sd["pis"].toIntVector();
    const IntVec opcs = sd["pcs"].toIntVector();
    const IntVec opidx = sd["path_indices"].toIntVector();
    const std::vector<double> opcoef = sd["path_coefficients"].toDoubleVector();
    int64_t obs = sd["batch_size"].toInt();
    const auto saved = ctx->get_saved_variables();
    const size_t ntens = saved.size();

    {
      at::TensorList sl(saved);
      auto r = handle_batch_dim_auto(obs, obatch, sl.slice(0, nin), sl.slice(nin));
      obs = r.first;
      obatch = r.second;
    }

    std::string bname;
    if (auto p = oname.find("_fwd"); p != std::string::npos) bname = oname.substr(0, p) + "_bwd" + oname.substr(p + 4);
    else if (auto q = oname.find("_bwd"); q != std::string::npos) bname = oname.substr(0, q) + "_bwd_bwd" + oname.substr(q + 4);
    else bname = oname + "_bwd";

    std::vector<bool> ng(ntens);
    for (size_t i = 0; i < ntens; ++i) ng[i] = ctx->needs_input_grad(i);  // edges exist only for tensor inputs

    const int64_t bnin = nin + nout;
    int64_t bnout = 0;
    for (size_t i = 0; i < ntens; ++i) bnout += ng[i];

    IntVec bbdim = obdim, bnseg = onseg, bbatch = obatch, bdtypes = odtypes;
    IntVec bindex(oindex.begin(), oindex.begin() + nin + nout);
    for (size_t i = 0; i < ntens; ++i) {
      if (!ng[i]) continue;
      bbdim.push_back(obdim[i]);
      bnseg.push_back(onseg[i]);
      bbatch.push_back(obatch[i]);
      bindex.push_back(oindex[i]);
      bdtypes.push_back(odtypes[i]);
    }
    bindex.insert(bindex.end(), oindex.begin() + nin + nout, oindex.end());

    std::vector<IntVec> oops;
    {
      size_t k = 0;
      for (int64_t i = 0; i < onops; ++i) {
        oops.emplace_back(ooperations.begin() + k, ooperations.begin() + k + onoperands[i]);
        k += onoperands[i];
      }
    }
    std::vector<IntVec> bops;
    IntVec bnpaths, bpis, bpcs;
    int64_t out_idx = bnin;
    for (size_t g = 0; g < ntens; ++g) {
      if (!ng[g]) continue;
      for (size_t oi = 0; oi < oops.size(); ++oi) {
        const auto& op = oops[oi];
        for (size_t pos = 0; pos < op.size(); ++pos) {
          if (op[pos] == (int64_t)g) {
            IntVec b = op;
            b[pos] = out_idx;
            bops.push_back(b);
            bnpaths.push_back(onpaths[oi]);
            bpis.push_back(opis[oi]);
            bpcs.push_back(opcs[oi]);
          }
        }
      }
      ++out_idx;
    }
    IntVec bnoperands, boperations;
    for (auto& o : bops) {
      bnoperands.push_back((int64_t)o.size());
      boperations.insert(boperations.end(), o.begin(), o.end());
    }

    std::vector<at::Tensor> btens(saved.begin(), saved.begin() + nin);
    for (auto& g : grad) btens.push_back(g);
    btens.insert(btens.end(), saved.begin() + nin, saved.end());

    std::vector<at::Tensor> bout;
    if (bnout > 0)
      bout = call_op(bname, math_dtype, ext, bnin, bnout, nidx, bbdim, bnseg, bbatch, bindex, bdtypes,
                     (int64_t)bops.size(), bnoperands, boperations, bnpaths, bpis, bpcs, opidx, opcoef, obs, btens);

    torch::autograd::variable_list res(kNumNonTensorArgs + ntens);
    size_t k = 0;
    for (size_t i = 0; i < ntens; ++i)
      if (ng[i]) res[kNumNonTensorArgs + i] = bout[k++];
    return res;
  }
};

std::vector<at::Tensor> uniform_1d_autograd(
    c10::string_view name, at::ScalarType math_dtype, int64_t operand_extent, int64_t num_inputs,
    int64_t num_outputs, int64_t num_index, c10::IntArrayRef buffer_dim, c10::IntArrayRef buffer_num_segments,
    c10::IntArrayRef batch_dim, c10::IntArrayRef index_buffer, c10::IntArrayRef dtypes, int64_t num_operations,
    c10::IntArrayRef num_operands, c10::IntArrayRef operations, c10::IntArrayRef num_paths,
    c10::IntArrayRef path_indices_start, c10::IntArrayRef path_coefficients_start, c10::IntArrayRef path_indices,
    c10::ArrayRef<double> path_coefficients, int64_t batch_size, at::TensorList tensors) {
  return Uniform1dFn::apply(name, math_dtype, operand_extent, num_inputs, num_outputs, num_index, buffer_dim,
                            buffer_num_segments, batch_dim, index_buffer, dtypes, num_operations, num_operands,
                            operations, num_paths, path_indices_start, path_coefficients_start, path_indices,
                            path_coefficients, batch_size, tensors);
}

}  // namespace

// Schema copied verbatim from torch.ops.cuequivariance.uniform_1d.default._schema (cuequivariance-ops-torch 0.11.1)
TORCH_LIBRARY(cuequivariance, m) {
  m.def(
      "uniform_1d(str name, ScalarType math_dtype, SymInt operand_extent, SymInt num_inputs, SymInt num_outputs, "
      "SymInt num_index, SymInt[] buffer_dim, SymInt[] buffer_num_segments, SymInt[] batch_dim, "
      "SymInt[] index_buffer, SymInt[] dtypes, SymInt num_operations, SymInt[] num_operands, SymInt[] operations, "
      "SymInt[] num_paths, SymInt[] path_indices_start, SymInt[] path_coefficients_start, SymInt[] path_indices, "
      "float[] path_coefficients, SymInt batch_size, Tensor[] tensors) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(cuequivariance, CUDA, m) { m.impl("uniform_1d", &uniform_1d_cuda); }
TORCH_LIBRARY_IMPL(cuequivariance, Autograd, m) { m.impl("uniform_1d", &uniform_1d_autograd); }
