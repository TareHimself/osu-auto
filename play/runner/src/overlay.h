#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <thread>
#include <vector>

#include "frame_ring.h"

struct OverlaySnapshot {
    bool active = false, focused = true;
    double box_x = 0.5, box_y = 0.5;
    double p_press = 0, p_down = 0;
    bool k1 = false, k2 = false;
    char key_names[2] = {'z', 'x'};
    long long presses = 0, k1_presses = 0, k2_presses = 0;
    double frame_age_ms = 0, infer_ms = 0, latency_ms = 0, capture_fps = 0;
    GrayFrame frame{};
    int attention_h = 0, attention_w = 0;
    std::vector<float> attention;
};

// Serves the overlay page from `dir` and streams snapshots to it as server-sent events on /events
class OverlayServer {
public:
    OverlayServer(int port, std::filesystem::path dir);
    ~OverlayServer();
    OverlayServer(const OverlayServer&) = delete;
    OverlayServer& operator=(const OverlayServer&) = delete;

    void publish(const OverlaySnapshot& snapshot);
    void set_active(bool active);

private:
    void serve();
    void handle(uintptr_t client);

    std::filesystem::path dir_;
    uintptr_t listener_;
    std::vector<uintptr_t> clients_;
    std::mutex mutex_;
    OverlaySnapshot snapshot_;
    uint64_t version_ = 0;
    std::atomic<bool> closing_{false};
    std::thread thread_;
};
