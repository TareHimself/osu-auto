#pragma once

#include <onnxruntime_cxx_api.h>

#include <array>
#include <chrono>
#include <string>
#include <vector>

#include "meta.h"

struct ModelOutput {
    double x, y, press_logit, down_logit;
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

        const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        inputs_[0] = Ort::Value::CreateTensor<float>(memory, frames_.data(), frames_.size(), frames_shape_.data(), frames_shape_.size());
        inputs_[1] = Ort::Value::CreateTensor<float>(memory, aux_.data(), aux_.size(), aux_shape_.data(), aux_shape_.size());
        for (int i = 0; i < 20; ++i) run(0.0f);
    }

    // The caller fills this before run()
    float* frames() { return frames_.data(); }

    ModelOutput run(float latency_aux, double* ms = nullptr) {
        aux_[0] = latency_aux;
        const auto t0 = std::chrono::steady_clock::now();
        auto out = session_.Run(Ort::RunOptions{nullptr}, input_names_, inputs_, 2, output_names_, 1);
        if (ms) *ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const float* o = out[0].GetTensorData<float>();
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
    const char* output_names_[1] = {"out"};
};
