#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(), 
           "Self Attention: all tensors must be contiguous");
    ASSERT(q->ndim() == 3, "Self Attention: q must be 3D [seqlen, nhead, d]");
    ASSERT(k->ndim() == 3, "Self Attention: k must be 3D [total_len, nkvhead, d]");
    ASSERT(v->ndim() == 3, "Self Attention: v must be 3D [total_len, nkvhead, dv]");
    ASSERT(attn_val->ndim() == 3, "Self Attention: attn_val must be 3D [seqlen, nhead, dv]");
    ASSERT(q->shape()[1] >= k->shape()[1], "Self Attention: q must have >= number of heads as k (for grouped-query attention)");
    ASSERT(k->shape()[1] == v->shape()[1], "Self Attention: k and v must have same number of heads");
    ASSERT(q->shape()[2] == k->shape()[2], "Self Attention: q and k must have same dimension");
    // Note: v can have different last dimension (dv) than k (d)
    ASSERT(attn_val->shape()[0] == q->shape()[0], "Self Attention: attn_val seqlen must match q seqlen");
    ASSERT(attn_val->shape()[1] == q->shape()[1], "Self Attention: attn_val nhead must match q nhead");
    ASSERT(attn_val->shape()[2] == v->shape()[2], "Self Attention: attn_val dv must match v dv");
    ASSERT(q->shape()[0] > 0 && q->shape()[1] > 0 && q->shape()[2] > 0, "Self Attention: q dimensions must be positive");
    ASSERT(k->shape()[0] > 0 && k->shape()[1] > 0 && k->shape()[2] > 0, "Self Attention: k dimensions must be positive");
    ASSERT(v->shape()[0] > 0 && v->shape()[1] > 0 && v->shape()[2] > 0, "Self Attention: v dimensions must be positive");
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());

    size_t seq_len = q->shape()[0];
    size_t nhead = q->shape()[1];
    size_t nkvhead = k->shape()[1];
    size_t d = q->shape()[2];
    size_t dv = v->shape()[2];
    size_t total_len = k->shape()[0];
    
    ASSERT(scale > 0.0f, "Self Attention: scale must be positive");
    ASSERT(total_len >= seq_len, "Self Attention: total_len must be >= seq_len");

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), 
                                  attn_val->dtype(), seq_len, nhead, nkvhead, d, dv, total_len, scale);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), 
                                  attn_val->dtype(), seq_len, nhead, nkvhead, d, dv, total_len, scale);
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
