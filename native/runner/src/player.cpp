#include "player.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "frame_ring.h"
#include "input.h"
#include "meta.h"
#include "model.h"
#include "npz.h"
#include "post.h"
#include "wgc_source.h"
#include "window.h"

namespace {

constexpr double kPeriod = 0.01;
constexpr size_t kTraceColumns = 9;

std::string stamp() {
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
    localtime_s(&tm, &t);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S", &tm);
    return buf;
}

double epoch_seconds() { return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count(); }

std::string json_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        out += c;
    }
    return out;
}

std::string narrow(const std::wstring& w) {
    std::string out(w.size(), '?');
    for (size_t i = 0; i < w.size(); ++i) {
        if (w[i] < 128) out[i] = static_cast<char>(w[i]);
    }
    return out;
}

// Wakes shortly before the deadline on a high resolution timer, then spins the rest
class Waiter {
public:
    Waiter() : timer_(CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS)) {}
    ~Waiter() {
        if (timer_) CloseHandle(timer_);
    }

    void until(double deadline) const {
        const double remaining = deadline - now_seconds();
        if (remaining > 0.002) {
            if (timer_) {
                LARGE_INTEGER due;
                due.QuadPart = -static_cast<LONGLONG>((remaining - 0.0015) * 1e7);
                SetWaitableTimer(timer_, &due, 0, nullptr, nullptr, FALSE);
                WaitForSingleObject(timer_, INFINITE);
            } else {
                Sleep(static_cast<DWORD>((remaining - 0.002) * 1000));
            }
        }
        while (now_seconds() < deadline) {
        }
    }

private:
    HANDLE timer_;
};

}  // namespace

struct Player::Impl {
    PlayOptions options;
    HWND hwnd;
    ModelMeta meta;
    std::vector<int> steps;
    FrameRing ring;
    std::unique_ptr<WgcSource> source;
    std::unique_ptr<Model> model;
    Injector injector;
    std::array<uint16_t, 2> scans;

    OneEuro smooth;
    KeyExecutor keys;
    std::mutex tick_mutex;
    mutable std::mutex stats_mutex;
    PlayStats stats;
    std::atomic<bool> active{false};
    std::atomic<bool> closing{false};
    std::thread loop;

    std::ofstream log;
    std::string trace_path;
    std::vector<double> trace;
    double last_infer_ms = 1.0;
    int held_key = -1;

    Impl(HWND hwnd_, const PlayOptions& o)
        : options(o), hwnd(hwnd_), smooth(kPeriod, o.smooth_cutoff, o.smooth_beta, o.smooth_dcut), keys(2, 0.5, 1.0) {
        std::string onnx = options.model;
        const auto dot = onnx.rfind('.');
        meta = load_meta(onnx.substr(0, dot) + ".json");
        steps = meta.steps();
        scans = {Injector::scan_code(options.keys[0]), Injector::scan_code(options.keys[1])};

        std::filesystem::create_directories(options.log_dir);
        log.open(options.log_dir + "/play-" + stamp() + ".jsonl", std::ios::app);

        SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
        model = std::make_unique<Model>(options.model, meta, options.cuda);
        source = std::make_unique<WgcSource>(hwnd, ring);

        char title[256]{};
        GetWindowTextA(hwnd, title, sizeof(title));
        const WindowGeometry g = source->geometry();
        char buf[1024];
        std::snprintf(buf, sizeof(buf),
                      "\"model\": \"%s\", \"window\": \"%s\", \"client\": [%d, %d, %d, %d], \"smooth_cutoff\": %g, \"smooth_beta\": %g, \"smooth_dcut\": %g, "
                      "\"latency_extra\": %g, \"latency_max\": %g, \"runner\": \"native\"",
                      json_escape(options.model).c_str(), json_escape(title).c_str(), g.client_x, g.client_y, g.client_w, g.client_h, options.smooth_cutoff, options.smooth_beta,
                      options.smooth_dcut, options.latency_extra_ms, meta.latency_max_ms);
        event("init", buf);
        loop = std::thread([this] { run(); });
    }

    ~Impl() {
        stop();
        closing = true;
        if (loop.joinable()) loop.join();
    }

    void event(const char* kind, const std::string& fields = "") {
        char head[96];
        std::snprintf(head, sizeof(head), "{\"t\": %.6f, \"event\": \"%s\"", epoch_seconds(), kind);
        log << head << (fields.empty() ? "" : ", ") << fields << "}\n";
        log.flush();
    }

    void start() {
        std::lock_guard lock(tick_mutex);
        smooth = OneEuro(kPeriod, options.smooth_cutoff, options.smooth_beta, options.smooth_dcut);
        keys = KeyExecutor(2, 0.5, 1.0);
        trace.clear();
        trace_path = options.log_dir + "/play-" + stamp() + "-trace.npz";
        {
            std::lock_guard stats_lock(stats_mutex);
            stats = PlayStats{};
        }
        active = true;
        char title[256]{};
        GetWindowTextA(hwnd, title, sizeof(title));
        event("start", std::string("\"window\": \"") + json_escape(title) + "\"");
    }

    void stop() {
        if (!active.exchange(false)) return;
        std::lock_guard lock(tick_mutex);
        release_held();
        long long presses, ticks;
        {
            std::lock_guard stats_lock(stats_mutex);
            presses = stats.presses;
            ticks = stats.ticks;
        }
        char buf[512];
        std::snprintf(buf, sizeof(buf), "\"presses\": %lld, \"ticks\": %lld, \"trace\": \"%s\"", presses, ticks, trace.empty() ? "" : json_escape(trace_path).c_str());
        event("stop", buf);
        if (!trace.empty()) {
            npz::write(trace_path, {{"trace.npy", npz::matrix_f64(trace, trace.size() / kTraceColumns, kTraceColumns)},
                                    {"columns.npy", npz::strings({"t", "x", "y", "p_press", "p_down", "k1", "k2", "frame_age_ms", "infer_ms"}, 12)}});
            trace.clear();
        }
    }

    void release_held() {
        if (held_key >= 0) injector.key(scans[held_key], false);
        held_key = -1;
    }

    void run() {
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
        const Waiter waiter;
        std::vector<double> loop_ms;
        double next_tick = now_seconds(), last_tick = next_tick, window_start = next_tick;
        double gap_max = 0, age_max = 0, infer_max = 0;
        double last_px = 0, last_py = 0, last_press = 0, last_down = 0;
        bool was_active = false;

        while (!closing) {
            if (!active) {
                was_active = false;
                Sleep(20);
                next_tick = now_seconds();
                continue;
            }
            if (!was_active) {
                was_active = true;
                next_tick = last_tick = window_start = now_seconds();
                gap_max = age_max = infer_max = 0;
                loop_ms.clear();
            }
            next_tick += kPeriod;
            const double t_start = now_seconds();
            gap_max = std::max(gap_max, (t_start - last_tick) * 1000);
            last_tick = t_start;

            if (t_start - window_start >= 1.0) {
                PlayStats s;
                {
                    std::lock_guard lock(stats_mutex);
                    stats.tick_gap_max_ms = gap_max;
                    stats.frame_age_max_ms = age_max;
                    stats.infer_max_ms = infer_max;
                    stats.capture_fps = source->fps();
                    s = stats;
                }
                char buf[768];
                std::snprintf(buf, sizeof(buf),
                              "\"active\": true, \"capture_fps\": %.3f, \"loop_ms\": %.4f, \"frame_age_ms\": %.3f, \"presses\": %lld, \"ticks\": %lld, "
                              "\"target\": [%.0f, %.0f], \"p_press\": %.5f, \"p_down\": %.5f, \"tick_gap_max_ms\": %.3f, \"frame_age_max_ms\": %.3f, "
                              "\"infer_max_ms\": %.3f, \"latency_ms\": %.3f",
                              s.capture_fps, s.loop_ms, s.frame_age_ms, s.presses, s.ticks, last_px, last_py, last_press, last_down, s.tick_gap_max_ms, s.frame_age_max_ms,
                              s.infer_max_ms, s.latency_ms);
                event("stats", buf);
                gap_max = age_max = infer_max = 0;
                window_start = t_start;
            }

            {
                std::lock_guard tick_lock(tick_mutex);
                if (active) tick(t_start, last_px, last_py, last_press, last_down, age_max, infer_max);
            }

            loop_ms.push_back((now_seconds() - t_start) * 1000);
            if (loop_ms.size() > 100) loop_ms.erase(loop_ms.begin());
            double mean = 0;
            for (double v : loop_ms) mean += v;
            {
                std::lock_guard lock(stats_mutex);
                stats.loop_ms = mean / static_cast<double>(loop_ms.size());
                ++stats.ticks;
            }

            const double now = now_seconds();
            if (next_tick < now - kPeriod) next_tick = now;
            waiter.until(next_tick);
        }
    }

    void tick(double t_start, double& last_px, double& last_py, double& last_press, double& last_down, double& age_max, double& infer_max) {
        const auto age = ring.stack_into(t_start, steps, kPeriod, model->frames());
        if (!age) return;

        double latency_ms = 0;
        float aux = 0.0f;
        if (meta.latency_max_ms > 0) {
            latency_ms = *age * 1000 + last_infer_ms + options.latency_extra_ms;
            aux = static_cast<float>(std::min(latency_ms, meta.latency_max_ms) / meta.latency_max_ms);
        }
        double infer_ms = 0;
        const ModelOutput o = model->run(aux, &infer_ms);
        last_infer_ms = infer_ms;
        infer_max = std::max(infer_max, infer_ms);

        const double p_press = 1.0 / (1.0 + std::exp(-o.press_logit)), p_down = 1.0 / (1.0 + std::exp(-o.down_logit));
        const auto [px, py] = smooth(decode_to_playfield(o.x, o.y));
        const auto [sx, sy] = playfield_to_screen(source->geometry(), px, py);
        const Keys k = keys.apply(p_press, p_down);
        const bool focused = GetForegroundWindow() == hwnd;
        if (options.inject) {
            if (focused) injector.move(sx, sy);
            const int want = focused && keys.down() ? keys.key() : -1;
            if (want != held_key) {
                release_held();
                if (want >= 0) injector.key(scans[want], true);
                held_key = want;
            }
        }

        age_max = std::max(age_max, *age * 1000);
        last_px = sx;
        last_py = sy;
        last_press = p_press;
        last_down = p_down;
        {
            std::lock_guard lock(stats_mutex);
            stats.presses += keys.pressed();
            stats.frame_age_ms = *age * 1000;
            stats.latency_ms = latency_ms;
            stats.focused = focused;
        }
        const double row[kTraceColumns] = {t_start, px, py, p_press, p_down, static_cast<double>(k.k1), static_cast<double>(k.k2), *age * 1000, infer_ms};
        trace.insert(trace.end(), row, row + kTraceColumns);
    }
};

Player::Player(HWND hwnd, const PlayOptions& options) : impl_(std::make_unique<Impl>(hwnd, options)) {}
Player::~Player() = default;
void Player::start() { impl_->start(); }
void Player::stop() { impl_->stop(); }
void Player::toggle() {
    if (impl_->active) {
        impl_->stop();
    } else {
        impl_->start();
    }
}
bool Player::active() const { return impl_->active; }
PlayStats Player::stats() const {
    std::lock_guard lock(impl_->stats_mutex);
    return impl_->stats;
}
