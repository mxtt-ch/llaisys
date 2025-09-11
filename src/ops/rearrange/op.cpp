#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rearrange_cpu.hpp"

namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    CHECK_SAME_DEVICE(out, in);
    ASSERT(out->isContiguous(), "Rearrange: output tensor must be contiguous");
    ASSERT(out->numel() == in->numel(), "Rearrange: output and input must have same number of elements");
    ASSERT(out->numel() > 0, "Rearrange: tensors must not be empty");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());

    size_t size = out->numel();

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rearrange(out->data(), in->data(), out->dtype(), size);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rearrange(out->data(), in->data(), out->dtype(), size);
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
