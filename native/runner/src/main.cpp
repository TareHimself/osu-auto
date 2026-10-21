#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "npy.h"
#include "post.h"

namespace {

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

int verify(const std::string& model_path, const std::string& dir, bool cuda) {
    const Npy frames = load_npy(dir + "/frames.npy");
    const Npy aux = load_npy(dir + "/aux.npy");
    const Npy out = load_npy(dir + "/out.npy");
    const Npy post = load_npy(dir + "/post.npy");
    const int64_t ticks = frames.shape[0], stack = frames.shape[1], h = frames.shape[2], w = frames.shape[3];
    const size_t frame_floats = static_cast<size_t>(stack * h * w);

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "runner");
    Ort::SessionOptions options;
    if (cuda) {
        Ort::CUDAProviderOptions cuda_options;
        cuda_options.Update({{"use_tf32", "0"}});
        options.AppendExecutionProvider_CUDA_V2(*cuda_options);
    }
    const std::wstring wide(model_path.begin(), model_path.end());
    Ort::Session session(env, wide.c_str(), options);

    const std::vector<int64_t> frames_shape = {1, stack, h, w};
    const std::vector<int64_t> aux_shape = {1, 2};
    const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const char* input_names[] = {"frames", "aux"};
    const char* output_names[] = {"out"};

    OneEuro smooth(0.01, 2.0, 0.05, 10.0);
    KeyExecutor keys(2, 0.5, 1.0);
    std::vector<float> x(frame_floats);
    double worst_out = 0, worst_aim = 0, total_ms = 0;
    int64_t key_mismatch = 0;

    for (int64_t t = 0; t < ticks; ++t) {
        const uint8_t* src = frames.as<uint8_t>() + static_cast<size_t>(t) * frame_floats;
        for (size_t i = 0; i < frame_floats; ++i) x[i] = static_cast<float>(src[i]) / 255.0f;
        std::vector<float> a = {aux.as<float>()[t], 0.0f};
        Ort::Value inputs[] = {
            Ort::Value::CreateTensor<float>(memory, x.data(), x.size(), frames_shape.data(), frames_shape.size()),
            Ort::Value::CreateTensor<float>(memory, a.data(), a.size(), aux_shape.data(), aux_shape.size()),
        };
        const auto t0 = std::chrono::steady_clock::now();
        auto result = session.Run(Ort::RunOptions{nullptr}, input_names, inputs, 2, output_names, 1);
        total_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

        const float* o = result[0].GetTensorData<float>();
        const double* ref = out.as<double>() + t * 4;
        worst_out = std::max({worst_out, std::abs(o[0] - ref[0]), std::abs(o[1] - ref[1]), std::abs(sigmoid(o[2]) - ref[2]), std::abs(sigmoid(o[3]) - ref[3])});

        // post-processing runs on the reference outputs, so model noise cannot flip a key edge
        const auto [px, py] = smooth(decode_to_playfield(ref[0], ref[1]));
        const Keys k = keys.apply(ref[2], ref[3]);
        const double* want = post.as<double>() + t * 4;
        worst_aim = std::max({worst_aim, std::abs(px - want[0]), std::abs(py - want[1])});
        if (static_cast<double>(k.k1) != want[2] || static_cast<double>(k.k2) != want[3]) ++key_mismatch;
    }

    std::printf("%s, %lld ticks, %.3f ms/tick\n", cuda ? "cuda" : "cpu", static_cast<long long>(ticks), total_ms / static_cast<double>(ticks));
    std::printf("model output max abs diff vs torch (x, y, p_press, p_down): %.2e\n", worst_out);
    std::printf("smoothed aim max abs diff vs python (playfield units): %.2e\n", worst_aim);
    std::printf("key state mismatches vs python: %lld of %lld\n", static_cast<long long>(key_mismatch), static_cast<long long>(ticks));
    return key_mismatch == 0 && worst_aim < 1e-6 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 4 && std::string(argv[1]) == "verify") {
        return verify(argv[2], argv[3], argc < 5 || std::string(argv[4]) == "cuda");
    }
    std::fprintf(stderr, "usage: runner verify <model.onnx> <fixture dir> [cpu|cuda]\n");
    return 2;
}
