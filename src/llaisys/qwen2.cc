#include "llaisys/models/qwen2.h"
#include "llaisys/tensor.h"
#include <vector>
#include <cstddef>

struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;
    LlaisysQwen2Weights weights;
    llaisysDeviceType_t device;
    std::vector<int> device_ids;
};

__C {

    __export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta,
                                                            llaisysDeviceType_t device,
                                                            int *device_ids,
                                                            int ndevice) {
        if (meta == nullptr) return nullptr;
        LlaisysQwen2Model *model = new LlaisysQwen2Model();
        model->meta = *meta;
        model->device = device;
        if (device_ids != nullptr && ndevice > 0) {
            model->device_ids.assign(device_ids, device_ids + ndevice);
        }
        model->weights = {};
        return model;
    }

    __export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model) {
        if (!model) return;
        delete model;
    }

    __export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model) {
        if (!model) return nullptr;
        return &model->weights;
    }

    __export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model,
                                            int64_t * token_ids,
                                            size_t ntoken) {
        if (!model)  return -1;
        return model->meta.end_token;
    }

}


