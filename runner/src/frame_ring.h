#pragma once

#include <windows.h>

#include <array>
#include <cmath>
#include <deque>
#include <mutex>
#include <optional>
#include <vector>

constexpr int kFrameW = 160;
constexpr int kFrameH = 120;

using GrayFrame = std::array<uint8_t, kFrameW * kFrameH>;

inline double qpc_to_seconds(int64_t ticks) {
    static const double inverse_freq = [] {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        return 1.0 / static_cast<double>(f.QuadPart);
    }();
    return static_cast<double>(ticks) * inverse_freq;
}

inline double now_seconds() {
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return qpc_to_seconds(c.QuadPart);
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

    // Fills dst with the frame nearest now - steps[k] * spacing for each k, scaled to 0..1, and returns the age of the newest frame
    std::optional<double> stack_into(double now, const std::vector<int>& steps, double spacing, float* dst) const {
        std::lock_guard lock(mutex_);
        if (items_.empty()) return std::nullopt;
        for (size_t k = 0; k < steps.size(); ++k) {
            const double want = now - steps[k] * spacing;
            const TimedFrame* best = &items_.front();
            double best_distance = std::abs(best->t - want);
            for (const auto& item : items_) {
                const double distance = std::abs(item.t - want);
                if (distance < best_distance) {
                    best = &item;
                    best_distance = distance;
                }
            }
            float* out = dst + k * best->pixels.size();
            for (size_t i = 0; i < best->pixels.size(); ++i) out[i] = static_cast<float>(best->pixels[i]) / 255.0f;
        }
        return now - items_.back().t;
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
