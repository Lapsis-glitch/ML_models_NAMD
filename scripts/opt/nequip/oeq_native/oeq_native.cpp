// Native (Python-free) build of OpenEquivariance's TorchScript ops for NAMD's libtorch shim.
//
// OpenEquivariance 0.7.0 registers the libtorch_tp_jit::* ops (schema + CUDA kernels) in C++
// (extension/torch_core.hpp), but their AUTOGRAD formulas live in Python
// (openequivariance/_torch/TensorProductConv.py: register_autograd), and its shipped .so files either
// embed pybind11 (JIT build) or target CUDA 12 (precompiled stable-ABI build).  This file compiles the
// upstream torch_core.hpp unchanged against the libtorch zip, drops the pybind11 module, and adds C++
// Autograd kernels that mirror the Python ones line by line:
//   jit_conv_forward  --backward-->  jit_conv_backward  --backward-->  jit_conv_double_backward
//   jit_tp_forward    --backward-->  jit_tp_backward    --backward-->  jit_tp_double_backward
// (the Python side also registers a triple backward for *_double_backward; not needed for forces or
// for force-derivatives, so it is left out -- calling it under autograd raises).
#include <cstdint>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/Operators.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <torch/all.h>
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>

using Tensor = torch::Tensor;
using Dtype = torch::Dtype;

constexpr Dtype kFloat = torch::kFloat;
constexpr Dtype kDouble = torch::kDouble;
constexpr Dtype kInt = torch::kInt;
constexpr Dtype kLong = torch::kLong;
constexpr Dtype kByte = torch::kByte;

#define TCHECK TORCH_CHECK
#define BOX(x) x
#define REGISTER_LIBRARY_IMPL TORCH_LIBRARY_IMPL
#define REGISTER_LIBRARY TORCH_LIBRARY

#include "torch_core.hpp"

// ---- helpers torch_core.hpp declares (verbatim from upstream libtorch_tp_jit.cpp) ----
Tensor tensor_to_cpu_contiguous(const Tensor &tensor) { return tensor.to(torch::kCPU).contiguous(); }
Tensor tensor_contiguous(const Tensor &tensor) { return tensor.contiguous(); }
Tensor tensor_empty_like(const Tensor &ref, const std::vector<int64_t> &sizes) {
    return torch::empty(sizes, ref.options());
}
Tensor tensor_zeros_like(const Tensor &ref, const std::vector<int64_t> &sizes) {
    return torch::zeros(sizes, ref.options());
}
void tensor_zero_(Tensor &tensor) { tensor.zero_(); }
void alert_not_deterministic(const char *name) { at::globalContext().alertNotDeterministic(name); }
const uint8_t *tensor_data_ptr_u8(const Tensor &tensor) { return tensor.data_ptr<uint8_t>(); }
void *data_ptr(const Tensor &tensor) {
    if (tensor.dtype() == torch::kFloat) return reinterpret_cast<void *>(tensor.data_ptr<float>());
    else if (tensor.dtype() == torch::kDouble) return reinterpret_cast<void *>(tensor.data_ptr<double>());
    else if (tensor.dtype() == torch::kLong) return reinterpret_cast<void *>(tensor.data_ptr<int64_t>());
    else if (tensor.dtype() == torch::kByte) return reinterpret_cast<void *>(tensor.data_ptr<uint8_t>());
    else if (tensor.dtype() == torch::kInt) return reinterpret_cast<void *>(tensor.data_ptr<int32_t>());
    else throw std::logic_error("Unsupported tensor datatype!");
}
Stream get_current_stream() { return c10::cuda::getCurrentCUDAStream(); }

// ---------------------------------------------------------------------------------------------
//  Autograd kernels
// ---------------------------------------------------------------------------------------------
using torch::autograd::AutogradContext;
using torch::autograd::variable_list;

namespace {

Tensor zero_if_undef(const Tensor &g, const Tensor &like) {
    return g.defined() ? g : torch::zeros_like(like);
}

template <typename Sig>
c10::TypedOperatorHandle<Sig> op(const char *name) {
    return c10::Dispatcher::singleton().findSchemaOrThrow(name, "").template typed<Sig>();
}

using ConvFwdSig = Tensor(Tensor, int64_t, Tensor, Tensor, Tensor, int64_t, Tensor, Tensor, Tensor, Tensor);
using ConvBwdSig = std::tuple<Tensor, Tensor, Tensor>(Tensor, int64_t, Tensor, Tensor, Tensor, Tensor, Tensor,
                                                      Tensor, Tensor, Tensor);
using ConvDBwdSig = std::tuple<Tensor, Tensor, Tensor, Tensor>(Tensor, int64_t, Tensor, Tensor, Tensor, Tensor,
                                                               Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
                                                               Tensor);
using TpFwdSig = Tensor(Tensor, int64_t, Tensor, Tensor, Tensor, int64_t);
using TpBwdSig = std::tuple<Tensor, Tensor, Tensor>(Tensor, int64_t, Tensor, Tensor, Tensor, Tensor);
using TpDBwdSig = std::tuple<Tensor, Tensor, Tensor, Tensor>(Tensor, int64_t, Tensor, Tensor, Tensor, Tensor,
                                                             Tensor, Tensor, Tensor);

// ---- convolution ----
struct ConvFwdFn : public torch::autograd::Function<ConvFwdFn> {
    static Tensor forward(AutogradContext *ctx, Tensor kernel, int64_t hash, Tensor L1, Tensor L2, Tensor W,
                          int64_t L3_dim, Tensor rows, Tensor cols, Tensor ws, Tensor perm) {
        ctx->save_for_backward({kernel, L1, L2, W, rows, cols, ws, perm});
        ctx->saved_data["hash"] = hash;
        at::AutoDispatchBelowADInplaceOrView guard;
        return op<ConvFwdSig>("libtorch_tp_jit::jit_conv_forward")
            .call(kernel, hash, L1, L2, W, L3_dim, rows, cols, ws, perm);
    }
    static variable_list backward(AutogradContext *ctx, variable_list go) {
        auto s = ctx->get_saved_variables();
        int64_t hash = ctx->saved_data["hash"].toInt();
        // Dispatch through the op again so its own Autograd kernel (below) records double backward.
        auto [g1, g2, gw] = op<ConvBwdSig>("libtorch_tp_jit::jit_conv_backward")
                                .call(s[0], hash, s[1], s[2], s[3], go[0].contiguous(), s[4], s[5], s[6], s[7]);
        return {Tensor(), Tensor(), g1, g2, gw, Tensor(), Tensor(), Tensor(), Tensor(), Tensor()};
    }
};

struct ConvBwdFn : public torch::autograd::Function<ConvBwdFn> {
    static variable_list forward(AutogradContext *ctx, Tensor kernel, int64_t hash, Tensor L1, Tensor L2, Tensor W,
                                 Tensor L3g, Tensor rows, Tensor cols, Tensor ws, Tensor perm) {
        ctx->save_for_backward({kernel, L1, L2, W, L3g, rows, cols, ws, perm});
        ctx->saved_data["hash"] = hash;
        at::AutoDispatchBelowADInplaceOrView guard;
        auto [a, b, c] = op<ConvBwdSig>("libtorch_tp_jit::jit_conv_backward")
                             .call(kernel, hash, L1, L2, W, L3g, rows, cols, ws, perm);
        return {a, b, c};
    }
    static variable_list backward(AutogradContext *ctx, variable_list g) {
        auto s = ctx->get_saved_variables();
        int64_t hash = ctx->saved_data["hash"].toInt();
        at::AutoDispatchBelowADInplaceOrView guard;  // no triple backward
        auto [r0, r1, r2, r3] = op<ConvDBwdSig>("libtorch_tp_jit::jit_conv_double_backward")
            .call(s[0], hash, s[1], s[2], s[3], s[4],
                  zero_if_undef(g[0], s[1]).contiguous(), zero_if_undef(g[1], s[2]).contiguous(),
                  zero_if_undef(g[2], s[3]).contiguous(), s[5], s[6], s[7], s[8]);
        return {Tensor(), Tensor(), r0, r1, r2, r3, Tensor(), Tensor(), Tensor(), Tensor()};
    }
};

// ---- plain (non-conv) tensor product, same pattern ----
struct TpFwdFn : public torch::autograd::Function<TpFwdFn> {
    static Tensor forward(AutogradContext *ctx, Tensor kernel, int64_t hash, Tensor L1, Tensor L2, Tensor W,
                          int64_t L3_dim) {
        ctx->save_for_backward({kernel, L1, L2, W});
        ctx->saved_data["hash"] = hash;
        at::AutoDispatchBelowADInplaceOrView guard;
        return op<TpFwdSig>("libtorch_tp_jit::jit_tp_forward").call(kernel, hash, L1, L2, W, L3_dim);
    }
    static variable_list backward(AutogradContext *ctx, variable_list go) {
        auto s = ctx->get_saved_variables();
        int64_t hash = ctx->saved_data["hash"].toInt();
        auto [g1, g2, gw] = op<TpBwdSig>("libtorch_tp_jit::jit_tp_backward")
                                .call(s[0], hash, s[1], s[2], s[3], go[0].contiguous());
        return {Tensor(), Tensor(), g1, g2, gw, Tensor()};
    }
};

struct TpBwdFn : public torch::autograd::Function<TpBwdFn> {
    static variable_list forward(AutogradContext *ctx, Tensor kernel, int64_t hash, Tensor L1, Tensor L2, Tensor W,
                                 Tensor L3g) {
        ctx->save_for_backward({kernel, L1, L2, W, L3g});
        ctx->saved_data["hash"] = hash;
        at::AutoDispatchBelowADInplaceOrView guard;
        auto [a, b, c] = op<TpBwdSig>("libtorch_tp_jit::jit_tp_backward").call(kernel, hash, L1, L2, W, L3g);
        return {a, b, c};
    }
    static variable_list backward(AutogradContext *ctx, variable_list g) {
        auto s = ctx->get_saved_variables();
        int64_t hash = ctx->saved_data["hash"].toInt();
        at::AutoDispatchBelowADInplaceOrView guard;
        auto [r0, r1, r2, r3] = op<TpDBwdSig>("libtorch_tp_jit::jit_tp_double_backward")
            .call(s[0], hash, s[1], s[2], s[3], s[4],
                  zero_if_undef(g[0], s[1]).contiguous(), zero_if_undef(g[1], s[2]).contiguous(),
                  zero_if_undef(g[2], s[3]).contiguous());
        return {Tensor(), Tensor(), r0, r1, r2, r3};
    }
};

Tensor conv_fwd_ag(Tensor kernel, int64_t hash, Tensor L1, Tensor L2, Tensor W, int64_t L3_dim, Tensor rows,
                   Tensor cols, Tensor ws, Tensor perm) {
    return ConvFwdFn::apply(kernel, hash, L1, L2, W, L3_dim, rows, cols, ws, perm);
}
std::tuple<Tensor, Tensor, Tensor> conv_bwd_ag(Tensor kernel, int64_t hash, Tensor L1, Tensor L2, Tensor W,
                                               Tensor L3g, Tensor rows, Tensor cols, Tensor ws, Tensor perm) {
    auto r = ConvBwdFn::apply(kernel, hash, L1, L2, W, L3g, rows, cols, ws, perm);
    return {r[0], r[1], r[2]};
}
Tensor tp_fwd_ag(Tensor kernel, int64_t hash, Tensor L1, Tensor L2, Tensor W, int64_t L3_dim) {
    return TpFwdFn::apply(kernel, hash, L1, L2, W, L3_dim);
}
std::tuple<Tensor, Tensor, Tensor> tp_bwd_ag(Tensor kernel, int64_t hash, Tensor L1, Tensor L2, Tensor W,
                                             Tensor L3g) {
    auto r = TpBwdFn::apply(kernel, hash, L1, L2, W, L3g);
    return {r[0], r[1], r[2]};
}

}  // namespace

TORCH_LIBRARY_IMPL(libtorch_tp_jit, Autograd, m) {
    m.impl("jit_conv_forward", &conv_fwd_ag);
    m.impl("jit_conv_backward", &conv_bwd_ag);
    m.impl("jit_tp_forward", &tp_fwd_ag);
    m.impl("jit_tp_backward", &tp_bwd_ag);
}
