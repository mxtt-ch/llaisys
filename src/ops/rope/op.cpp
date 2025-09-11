#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(), 
           "RoPE: all tensors must be contiguous");
    ASSERT(in->ndim() == 3, "RoPE: input must be 3D [seqlen, nhead, d]");
    ASSERT(out->ndim() == 3, "RoPE: output must be 3D [seqlen, nhead, d]");
    ASSERT(pos_ids->ndim() == 1, "RoPE: pos_ids must be 1D");
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "RoPE: pos_ids must be int64");
    ASSERT(out->shape()[0] == in->shape()[0], "RoPE: output seqlen must match input seqlen");
    ASSERT(out->shape()[1] == in->shape()[1], "RoPE: output nhead must match input nhead");
    ASSERT(out->shape()[2] == in->shape()[2], "RoPE: output d must match input d");
    ASSERT(pos_ids->shape()[0] == in->shape()[0], "RoPE: pos_ids length must match seqlen");
    ASSERT(in->shape()[0] > 0 && in->shape()[1] > 0 && in->shape()[2] > 0, "RoPE: input dimensions must be positive");
    ASSERT(pos_ids->shape()[0] > 0, "RoPE: pos_ids length must be positive");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());

    size_t seq_len = in->shape()[0];
    size_t nhead = in->shape()[1];
    size_t d = in->shape()[2];
    
    ASSERT(theta > 0.0f, "RoPE: theta must be positive");
    ASSERT(d % 2 == 0, "RoPE: dimension must be even");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(), 
                        out->dtype(), seq_len, nhead, d, theta);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(), 
                        out->dtype(), seq_len, nhead, d, theta);
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
