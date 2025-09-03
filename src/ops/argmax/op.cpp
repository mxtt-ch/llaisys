#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/argmax_cpu.hpp"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    ASSERT(vals->isContiguous(), "Argmax: input tensor must be contiguous");
    ASSERT(max_idx->isContiguous() && max_val->isContiguous(), "Argmax: output tensors must be contiguous");
    ASSERT(vals->ndim() == 1, "Argmax: input tensor must be 1D");
    ASSERT(vals->numel() > 0, "Argmax: input tensor must not be empty");
    ASSERT(max_idx->numel() == 1 && max_val->numel() == 1, "Argmax: output tensors must have size 1");
    ASSERT(max_idx->dtype() == LLAISYS_DTYPE_I64, "Argmax: max_idx must be int64");
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());

    if (max_idx->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
    }

    llaisys::core::context().setDevice(max_idx->deviceType(), max_idx->deviceId());

    switch (max_idx->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
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
