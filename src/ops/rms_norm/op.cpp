#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), 
           "RMS Norm: all tensors must be contiguous");
    ASSERT(in->ndim() == 2, "RMS Norm: input must be 2D");
    ASSERT(out->ndim() == 2, "RMS Norm: output must be 2D");
    ASSERT(weight->ndim() == 1, "RMS Norm: weight must be 1D");
    ASSERT(out->shape()[0] == in->shape()[0], "RMS Norm: output batch size must match input batch size");
    ASSERT(out->shape()[1] == in->shape()[1], "RMS Norm: output hidden size must match input hidden size");
    ASSERT(weight->shape()[0] == in->shape()[1], "RMS Norm: weight size must match hidden size");
    ASSERT(in->shape()[0] > 0 && in->shape()[1] > 0, "RMS Norm: input dimensions must be positive");
    ASSERT(weight->shape()[0] > 0, "RMS Norm: weight size must be positive");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());

    size_t batch_size = in->shape()[0];
    size_t hidden_size = in->shape()[1];
    
    ASSERT(eps > 0.0f, "RMS Norm: eps must be positive");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(), 
                            out->dtype(), batch_size, hidden_size, eps);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(), 
                            out->dtype(), batch_size, hidden_size, eps);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
