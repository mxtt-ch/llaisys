#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/swiglu_cpu.hpp"

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(), 
           "SwiGLU: all tensors must be contiguous");
    ASSERT(out->ndim() == 2, "SwiGLU: output must be 2D");
    ASSERT(gate->ndim() == 2, "SwiGLU: gate must be 2D");
    ASSERT(up->ndim() == 2, "SwiGLU: up must be 2D");
    ASSERT(out->shape()[0] == gate->shape()[0], "SwiGLU: output first dim must match gate first dim");
    ASSERT(out->shape()[1] == gate->shape()[1], "SwiGLU: output second dim must match gate second dim");
    ASSERT(out->shape()[0] == up->shape()[0], "SwiGLU: output first dim must match up first dim");
    ASSERT(out->shape()[1] == up->shape()[1], "SwiGLU: output second dim must match up second dim");
    ASSERT(out->shape()[0] > 0 && out->shape()[1] > 0, "SwiGLU: output dimensions must be positive");
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());

    size_t size = out->numel();

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::swiglu(out->data(), gate->data(), up->data(), out->dtype(), size);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::swiglu(out->data(), gate->data(), up->data(), out->dtype(), size);
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
