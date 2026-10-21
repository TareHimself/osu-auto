#include <onnxruntime_cxx_api.h>

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: runner <model.onnx> [cpu|cuda]\n");
        return 2;
    }
    const std::wstring model(argv[1], argv[1] + std::string(argv[1]).size());
    const bool cuda = argc < 3 || std::string(argv[2]) == "cuda";

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "runner");
    Ort::SessionOptions options;
    if (cuda) {
        OrtCUDAProviderOptions cuda_options{};
        options.AppendExecutionProvider_CUDA(cuda_options);
    }
    Ort::Session session(env, model.c_str(), options);

    std::vector<float> frames(1 * 4 * 120 * 160, 0.0f);
    std::vector<float> aux = {0.5f, 0.0f};
    const std::vector<int64_t> frames_shape = {1, 4, 120, 160};
    const std::vector<int64_t> aux_shape = {1, 2};
    const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputs[] = {
        Ort::Value::CreateTensor<float>(memory, frames.data(), frames.size(), frames_shape.data(), frames_shape.size()),
        Ort::Value::CreateTensor<float>(memory, aux.data(), aux.size(), aux_shape.data(), aux_shape.size()),
    };
    const char* input_names[] = {"frames", "aux"};
    const char* output_names[] = {"out"};

    std::vector<Ort::Value> out;
    for (int i = 0; i < 20; ++i) {
        out = session.Run(Ort::RunOptions{nullptr}, input_names, inputs, 2, output_names, 1);
    }
    const auto t0 = std::chrono::steady_clock::now();
    constexpr int runs = 200;
    for (int i = 0; i < runs; ++i) {
        out = session.Run(Ort::RunOptions{nullptr}, input_names, inputs, 2, output_names, 1);
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / runs;

    const float* o = out[0].GetTensorData<float>();
    std::printf("%s: %.3f ms/run, out = %.5f %.5f %.5f %.5f\n", cuda ? "cuda" : "cpu", ms, o[0], o[1], o[2], o[3]);
    return 0;
}
