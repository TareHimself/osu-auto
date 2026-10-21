#include <onnxruntime_cxx_api.h>
#include <winrt/base.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <fstream>
#include <thread>

#include "frame_ring.h"
#include "npy.h"
#include "post.h"
#include "processor.h"
#include "wgc_source.h"
#include "window.h"

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

int shader_test(const std::string& dir) {
    const Npy frame = load_npy(dir + "/frame.npy");
    const Npy expected = load_npy(dir + "/expected.npy");
    const Npy crop_npy = load_npy(dir + "/crop.npy");
    const int32_t* c = crop_npy.as<int32_t>();
    const Crop crop{c[0], c[1], c[2], c[3]};

    const Device device = make_device();
    GpuProcessor processor(device);
    D3D11_TEXTURE2D_DESC td{};
    td.Width = static_cast<UINT>(frame.shape[1]);
    td.Height = static_cast<UINT>(frame.shape[0]);
    td.MipLevels = 1;
    td.ArraySize = 1;
    td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    td.SampleDesc.Count = 1;
    td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA init{frame.data.data(), td.Width * 4, 0};
    Microsoft::WRL::ComPtr<ID3D11Texture2D> texture;
    if (FAILED(device.device->CreateTexture2D(&td, &init, &texture))) throw std::runtime_error("CreateTexture2D");

    GrayFrame out;
    processor.process(texture.Get(), crop, out);
    const uint8_t* want = expected.as<uint8_t>();
    int worst = 0;
    double sum = 0;
    int off_by_more_than_one = 0;
    for (size_t i = 0; i < out.size(); ++i) {
        const int d = std::abs(static_cast<int>(out[i]) - static_cast<int>(want[i]));
        worst = std::max(worst, d);
        sum += d;
        off_by_more_than_one += d > 1;
    }
    std::printf("crop %d,%d %dx%d of %lldx%lld: mean abs diff %.4f, max %d, pixels off by more than 1: %d\n", crop.x, crop.y, crop.w, crop.h,
                static_cast<long long>(frame.shape[1]), static_cast<long long>(frame.shape[0]), sum / static_cast<double>(out.size()), worst, off_by_more_than_one);
    return 0;
}

int capture(const std::wstring& prefix, int seconds, const char* pgm_path) {
    winrt::init_apartment(winrt::apartment_type::multi_threaded);
    const auto hwnd = find_window(prefix);
    if (!hwnd) {
        std::fprintf(stderr, "no window starting with the given title\n");
        return 1;
    }
    const WindowGeometry g = window_geometry(*hwnd);
    std::printf("client %dx%d, box %d,%d %dx%d\n", g.client_w, g.client_h, g.box.x, g.box.y, g.box.w, g.box.h);

    FrameRing ring;
    WgcSource source(*hwnd, ring);
    uint64_t last = 0;
    for (int i = 0; i < seconds; ++i) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        const uint64_t n = ring.count();
        const auto frame = ring.latest();
        std::printf("frames %llu (+%llu), capture %.0f fps, newest age %.1f ms\n", static_cast<unsigned long long>(n), static_cast<unsigned long long>(n - last), source.fps(),
                    frame ? (now_seconds() - frame->t) * 1000.0 : -1.0);
        last = n;
    }
    if (const auto frame = ring.latest(); frame && pgm_path) {
        std::ofstream pgm(pgm_path, std::ios::binary);
        pgm << "P5\n" << kFrameW << " " << kFrameH << "\n255\n";
        pgm.write(reinterpret_cast<const char*>(frame->pixels.data()), static_cast<std::streamsize>(frame->pixels.size()));
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 4 && std::string(argv[1]) == "verify") {
        return verify(argv[2], argv[3], argc < 5 || std::string(argv[4]) == "cuda");
    }
    if (argc >= 3 && std::string(argv[1]) == "shader-test") return shader_test(argv[2]);
    if (argc >= 4 && std::string(argv[1]) == "capture") {
        return capture(std::wstring(argv[2], argv[2] + std::string(argv[2]).size()), std::atoi(argv[3]), argc > 4 ? argv[4] : nullptr);
    }
    std::fprintf(stderr, "usage: runner verify <model.onnx> <fixture dir> [cpu|cuda]\n       runner shader-test <dir>\n       runner capture <title prefix> <seconds> [out.pgm]\n");
    return 2;
}
