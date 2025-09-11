#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(), 
           "Embedding: all tensors must be contiguous");
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding: index must be int64");
    ASSERT(index->ndim() == 1, "Embedding: index must be 1D");
    ASSERT(weight->ndim() == 2, "Embedding: weight must be 2D");
    ASSERT(out->ndim() == 2, "Embedding: output must be 2D");
    ASSERT(out->shape()[0] == index->shape()[0], "Embedding: output first dim must match index size");
    ASSERT(out->shape()[1] == weight->shape()[1], "Embedding: output second dim must match weight hidden size");
    ASSERT(index->shape()[0] > 0, "Embedding: index size must be positive");
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());

    size_t batch_size = index->shape()[0];
    size_t seq_len = 1;
    size_t hidden_size = weight->shape()[1];

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(), 
                             out->dtype(), weight->dtype(), batch_size, seq_len, hidden_size);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(), 
                             out->dtype(), weight->dtype(), batch_size, seq_len, hidden_size);
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
