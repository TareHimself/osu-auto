#pragma once

#include <windows.h>

#include <atomic>
#include <memory>
#include <string>

struct PlayOptions {
    std::string model;
    std::wstring window_prefix = L"osu!";
    bool inject = true;
    bool cuda = true;
    bool hook = false;
    double smooth_cutoff = 2.0;
    double smooth_beta = 0.05;
    double smooth_dcut = 10.0;
    double latency_extra_ms = 8.0;
    std::string keys = "zx";
    std::string log_dir = "data/play_logs";
};

struct PlayStats {
    double capture_fps = 0, loop_ms = 0, frame_age_ms = 0, latency_ms = 0;
    double tick_gap_max_ms = 0, frame_age_max_ms = 0, infer_max_ms = 0;
    long long presses = 0, ticks = 0;
    bool focused = true;
};

// Runs the model against one window at 100 Hz: capture, infer, smooth, move the cursor, press keys
class Player {
public:
    Player(HWND hwnd, const PlayOptions& options);
    ~Player();
    void start();
    void stop();
    void toggle();
    bool active() const;
    PlayStats stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
