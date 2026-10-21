#pragma once

#include <windows.h>

#include <array>
#include <deque>
#include <mutex>
#include <optional>

constexpr int kFrameW = 160;
constexpr int kFrameH = 120;

using GrayFrame = std::array<uint8_t, kFrameW * kFrameH>;

inline double now_seconds() {
    static const double inverse_freq = [] {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        return 1.0 / static_cast<double>(f.QuadPart);
    }();
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return static_cast<double>(c.QuadPart) * inverse_freq;
}

struct TimedFrame {
    double t;
    GrayFrame pixels;
};

class FrameRing {
public:
    void push(double t, const GrayFrame& pixels) {
        std::lock_guard lock(mutex_);
        if (items_.size() == kMax) items_.pop_front();
        items_.push_back({t, pixels});
        ++count_;
    }

    std::optional<TimedFrame> latest() const {
        std::lock_guard lock(mutex_);
        if (items_.empty()) return std::nullopt;
        return items_.back();
    }

    uint64_t count() const {
        std::lock_guard lock(mutex_);
        return count_;
    }

private:
    static constexpr size_t kMax = 128;
    mutable std::mutex mutex_;
    std::deque<TimedFrame> items_;
    uint64_t count_ = 0;
};
