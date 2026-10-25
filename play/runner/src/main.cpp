#include <onnxruntime_cxx_api.h>
#include <winrt/base.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iterator>
#include <string>
#include <vector>

#include <fstream>
#include <thread>

#include "frame_ring.h"
#include "frame_source.h"
#include "hook_inject.h"
#include "hook_source.h"
#include "npy.h"
#include "player.h"
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

    KeyExecutor keys(2, 0.5, 1.0);
    std::vector<float> x(frame_floats);
    double worst_out = 0, worst_aim = 0, total_ms = 0;
    int64_t key_mismatch = 0;

    {
        // The first runs pay for cuDNN's setup, which is not what a tick costs
        std::vector<float> zeros(frame_floats, 0.0f), zero_aux = {0.0f, 0.0f};
        Ort::Value warm[] = {
            Ort::Value::CreateTensor<float>(memory, zeros.data(), zeros.size(), frames_shape.data(), frames_shape.size()),
            Ort::Value::CreateTensor<float>(memory, zero_aux.data(), zero_aux.size(), aux_shape.data(), aux_shape.size()),
        };
        for (int i = 0; i < 20; ++i) session.Run(Ort::RunOptions{nullptr}, input_names, warm, 2, output_names, 1);
    }

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
        const auto [px, py] = decode_to_playfield(ref[0], ref[1]);
        const Keys k = keys.apply(ref[2], ref[3]);
        const double* want = post.as<double>() + t * 4;
        worst_aim = std::max({worst_aim, std::abs(px - want[0]), std::abs(py - want[1])});
        if (static_cast<double>(k.k1) != want[2] || static_cast<double>(k.k2) != want[3]) ++key_mismatch;
    }

    std::printf("%s, %lld ticks, %.3f ms/tick\n", cuda ? "cuda" : "cpu", static_cast<long long>(ticks), total_ms / static_cast<double>(ticks));
    std::printf("model output max abs diff vs torch (x, y, p_press, p_down): %.2e\n", worst_out);
    std::printf("aim max abs diff vs python (playfield units): %.2e\n", worst_aim);
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

int hook_command(const std::string& action, const std::wstring& prefix) {
    const auto hwnd = find_window(prefix);
    if (!hwnd) {
        std::fprintf(stderr, "no window starting with the given title\n");
        return 1;
    }
    const std::string error = action == "inject" ? inject_hook(*hwnd) : eject_hook(*hwnd);
    if (!error.empty()) {
        std::fprintf(stderr, "%s failed: %s\n", action.c_str(), error.c_str());
        return 1;
    }
    std::printf("%s ok\n", action.c_str());
    return 0;
}

int capture(const std::wstring& prefix, int seconds, const char* pgm_path, bool hook = false) {
    winrt::init_apartment(winrt::apartment_type::multi_threaded);
    const auto hwnd = find_window(prefix);
    if (!hwnd) {
        std::fprintf(stderr, "no window starting with the given title\n");
        return 1;
    }
    const WindowGeometry g = window_geometry(*hwnd);
    std::printf("client %dx%d, box %d,%d %dx%d\n", g.client_w, g.client_h, g.box.x, g.box.y, g.box.w, g.box.h);

    FrameRing ring;
    std::unique_ptr<FrameSource> source;
    if (hook) {
        source = std::make_unique<HookSource>(*hwnd, ring);
    } else {
        source = std::make_unique<WgcSource>(*hwnd, ring);
    }
    uint64_t last = 0;
    for (int i = 0; i < seconds; ++i) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        const uint64_t n = ring.count();
        const auto frame = ring.latest();
        std::printf("frames %llu (+%llu), capture %.0f fps, newest age %.1f ms\n", static_cast<unsigned long long>(n), static_cast<unsigned long long>(n - last), source->fps(),
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


const char* kDefaultModel = "data/models/current/joint_best.onnx";

std::filesystem::path find_default_model() {
    wchar_t buffer[MAX_PATH];
    const DWORD n = GetModuleFileNameW(nullptr, buffer, MAX_PATH);
    std::vector<std::filesystem::path> starts = {std::filesystem::current_path()};
    if (n > 0) starts.insert(starts.begin(), std::filesystem::path(buffer, buffer + n).parent_path());
    for (auto dir : starts) {
        for (; !dir.empty(); dir = dir.parent_path()) {
            const auto candidate = dir / kDefaultModel;
            if (std::filesystem::exists(candidate)) return candidate;
            if (dir == dir.parent_path()) break;
        }
    }
    return {};
}

void print_model(const std::string& model, bool is_default) {
    std::string label = std::filesystem::path(model).parent_path().filename().string();
    std::ifstream source(std::filesystem::path(model).parent_path() / "source.json");
    std::string text((std::istreambuf_iterator<char>(source)), std::istreambuf_iterator<char>());
    const auto key = text.find("\"run\"");
    if (key != std::string::npos) {
        const auto open = text.find('"', text.find(':', key) + 1);
        const auto close = open == std::string::npos ? open : text.find('"', open + 1);
        if (close != std::string::npos) label += " (" + text.substr(open + 1, close - open - 1) + ")";
    }
    std::printf("model: %s%s\n", label.c_str(), is_default ? "" : (" [" + model + "]").c_str());
}

int play(int argc, char** argv) {
    winrt::init_apartment(winrt::apartment_type::multi_threaded);
    PlayOptions options;
    const bool is_default = argc < 3 || std::string(argv[2]).rfind("--", 0) == 0;
    int first_option = is_default ? 2 : 3;
    if (is_default) {
        const auto found = find_default_model();
        if (found.empty()) {
            std::fprintf(stderr, "no model given and %s not found\n", kDefaultModel);
            return 1;
        }
        options.model = found.string();
    } else {
        options.model = argv[2];
    }
    print_model(options.model, is_default);
    int duration = 0;
    bool start_now = false;
    for (int i = first_option; i < argc; ++i) {
        const std::string a = argv[i];
        const bool has_value = i + 1 < argc;
        if (a == "--no-inject") {
            options.inject = false;
        } else if (a == "--start-now") {
            start_now = true;
        } else if (a == "--capture" && has_value) {
            options.hook = std::string(argv[++i]) == "hook";
        } else if (a == "--cpu") {
            options.cuda = false;
        } else if (a == "--window" && has_value) {
            const std::string v = argv[++i];
            options.window_prefix.assign(v.begin(), v.end());
        } else if (a == "--duration" && has_value) {
            duration = std::atoi(argv[++i]);
        } else if (a == "--latency-extra" && has_value) {
            options.latency_extra_ms = std::atof(argv[++i]);
        } else if (a == "--keys" && has_value) {
            options.keys = argv[++i];
        } else if (a == "--log-dir" && has_value) {
            options.log_dir = argv[++i];
        } else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }
    const auto hwnd = find_window(options.window_prefix);
    if (!hwnd) {
        std::fprintf(stderr, "no window starting with the given title\n");
        return 1;
    }
    Player player(*hwnd, options);
    constexpr int kToggle = 1, kStop = 2;
    RegisterHotKey(nullptr, kToggle, MOD_NOREPEAT, VK_F8);
    RegisterHotKey(nullptr, kStop, MOD_NOREPEAT, VK_F9);
    SetTimer(nullptr, 1, 1000, nullptr);
    std::printf("F8 starts and stops the agent, F9 quits. Input is only sent while the game window is focused.\n");
    if (start_now) player.start();

    int elapsed = 0;
    MSG msg;
    while (GetMessage(&msg, nullptr, 0, 0) > 0) {
        if (msg.message == WM_HOTKEY && msg.wParam == kToggle) {
            player.toggle();
            std::printf("agent %s\n", player.active() ? "on" : "off");
        } else if (msg.message == WM_HOTKEY && msg.wParam == kStop) {
            break;
        } else if (msg.message == WM_TIMER) {
            if (player.active()) {
                const PlayStats s = player.stats();
                std::printf("capture %.0f fps | frame age %.1f ms | loop %.2f ms | ticks %lld | presses %lld | stall max tick %.0f frame %.0f infer %.0f ms\n", s.capture_fps,
                            s.frame_age_ms, s.loop_ms, s.ticks, s.presses, s.tick_gap_max_ms, s.frame_age_max_ms, s.infer_max_ms);
            }
            if (player.active() && !player.stats().focused && options.inject) std::printf("game window not focused, input paused\n");
            if (duration > 0 && player.active() && ++elapsed >= duration) break;
        }
    }
    player.stop();
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 4 && std::string(argv[1]) == "verify") {
        return verify(argv[2], argv[3], argc < 5 || std::string(argv[4]) == "cuda");
    }
    if (argc >= 2 && std::string(argv[1]) == "play") return play(argc, argv);
    if (argc >= 3 && std::string(argv[1]) == "shader-test") return shader_test(argv[2]);
    if (argc >= 3 && std::string(argv[1]) == "hook") {
        const std::string prefix = argc > 3 ? argv[3] : "osu!";
        return hook_command(argv[2], std::wstring(prefix.begin(), prefix.end()));
    }
    if (argc >= 4 && std::string(argv[1]) == "hook-capture") {
        return capture(std::wstring(argv[2], argv[2] + std::string(argv[2]).size()), std::atoi(argv[3]), argc > 4 ? argv[4] : nullptr, true);
    }
    if (argc >= 4 && std::string(argv[1]) == "capture") {
        return capture(std::wstring(argv[2], argv[2] + std::string(argv[2]).size()), std::atoi(argv[3]), argc > 4 ? argv[4] : nullptr);
    }
    std::fprintf(stderr, "usage: runner verify <model.onnx> <fixture dir> [cpu|cuda]\n       runner shader-test <dir>\n       runner capture <title prefix> <seconds> [out.pgm]\n       runner play [model.onnx] [--window P] [--no-inject] [--cpu] [--start-now] [--duration S] [--keys zx] [--log-dir D] [--latency-extra MS]\n");
    return 2;
}
