#pragma once

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <string>
#include <vector>

#include "meta.h"

struct ModelOutput {
    double x, y, press_logit, down_logit;
};

struct AttentionShape {
    int h = 0, w = 0;
};

class Model {
public:
    Model(const std::string& onnx_path, const ModelMeta& meta, bool cuda)
        : env_(ORT_LOGGING_LEVEL_WARNING, "runner"),
          frames_(static_cast<size_t>(meta.stack) * meta.frame_h * meta.frame_w, 0.0f),
          frames_shape_{1, meta.stack, meta.frame_h, meta.frame_w} {
        if (cuda) {
            Ort::CUDAProviderOptions cuda_options;
            cuda_options.Update({{"use_tf32", "0"}});
            options_.AppendExecutionProvider_CUDA_V2(*cuda_options);
        }
        const std::wstring wide(onnx_path.begin(), onnx_path.end());
        session_ = Ort::Session(env_, wide.c_str(), options_);
        if (session_.GetOutputCount() > 1) {
            Ort::AllocatorWithDefaultOptions allocator;
            const auto shape = session_.GetOutputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();
            const bool named = std::string(session_.GetOutputNameAllocated(1, allocator).get()) == "attention";
            if (named && shape.size() >= 2 && shape[shape.size() - 2] > 0 && shape[shape.size() - 1] > 0) {
                attention_shape_ = {static_cast<int>(shape[shape.size() - 2]), static_cast<int>(shape[shape.size() - 1])};
                attention_.resize(static_cast<size_t>(attention_shape_.h) * attention_shape_.w);
                output_count_ = 2;
            }
        }

        const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        inputs_[0] = Ort::Value::CreateTensor<float>(memory, frames_.data(), frames_.size(), frames_shape_.data(), frames_shape_.size());
        inputs_[1] = Ort::Value::CreateTensor<float>(memory, aux_.data(), aux_.size(), aux_shape_.data(), aux_shape_.size());
        for (int i = 0; i < 20; ++i) run(0.0f);
    }

    // The caller fills this before run()
    float* frames() { return frames_.data(); }
    const float* frames() const { return frames_.data(); }

    const std::vector<float>& attention() const { return attention_; }
    AttentionShape attention_shape() const { return attention_shape_; }

    ModelOutput run(float latency_aux, double* ms = nullptr) {
        aux_[0] = latency_aux;
        const auto t0 = std::chrono::steady_clock::now();
        auto out = session_.Run(Ort::RunOptions{nullptr}, input_names_, inputs_, 2, output_names_, output_count_);
        if (ms) *ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const float* o = out[0].GetTensorData<float>();
        if (output_count_ > 1) {
            const float* a = out[1].GetTensorData<float>();
            std::copy(a, a + attention_.size(), attention_.begin());
        }
        return {o[0], o[1], o[2], o[3]};
    }

private:
    Ort::Env env_;
    Ort::SessionOptions options_;
    Ort::Session session_{nullptr};
    std::vector<float> frames_;
    std::array<float, 2> aux_{0.0f, 0.0f};
    std::array<int64_t, 4> frames_shape_;
    std::array<int64_t, 2> aux_shape_{1, 2};
    Ort::Value inputs_[2] = {Ort::Value(nullptr), Ort::Value(nullptr)};
    const char* input_names_[2] = {"frames", "aux"};
    const char* output_names_[2] = {"out", "attention"};
    size_t output_count_ = 1;
    std::vector<float> attention_;
    AttentionShape attention_shape_;
};
