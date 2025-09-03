#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), 
           "Linear: all tensors must be contiguous");
    ASSERT(in->ndim() == 2, "Linear: input must be 2D");
    ASSERT(weight->ndim() == 2, "Linear: weight must be 2D");
    ASSERT(out->ndim() == 2, "Linear: output must be 2D");
    ASSERT(in->shape()[1] == weight->shape()[1], "Linear: input features must match weight input features");
    ASSERT(out->shape()[0] == in->shape()[0], "Linear: output batch size must match input batch size");
    ASSERT(out->shape()[1] == weight->shape()[0], "Linear: output features must match weight output features");
    ASSERT(in->shape()[0] > 0 && in->shape()[1] > 0, "Linear: input dimensions must be positive");
    ASSERT(weight->shape()[0] > 0 && weight->shape()[1] > 0, "Linear: weight dimensions must be positive");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());

    size_t batch_size = in->shape()[0];
    size_t in_features = in->shape()[1];
    size_t out_features = weight->shape()[0];
    bool has_bias = (bias != nullptr);

    if (has_bias) {
        CHECK_SAME_DEVICE(out, bias);
        ASSERT(bias->isContiguous(), "Linear: bias must be contiguous");
        ASSERT(bias->ndim() == 1, "Linear: bias must be 1D");
        ASSERT(bias->shape()[0] == out_features, "Linear: bias size must match output features");
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
    }

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), 
                          has_bias ? bias->data() : nullptr,
                          out->dtype(), batch_size, in_features, out_features, has_bias);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), 
                          has_bias ? bias->data() : nullptr,
                          out->dtype(), batch_size, in_features, out_features, has_bias);
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
