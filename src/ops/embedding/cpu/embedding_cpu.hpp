#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, 
               llaisysDataType_t out_type, llaisysDataType_t weight_type,
               size_t batch_size, size_t seq_len, size_t hidden_size);
}
