#pragma once

#include <cmath>
#include <limits>
#include <numbers>
#include <utility>

constexpr double kPlayfieldW = 512.0;
constexpr double kPlayfieldH = 384.0;
constexpr double kAimLo = 0.25;
constexpr double kAimHi = 0.75;

inline std::pair<double, double> decode_to_playfield(double nx, double ny) {
    return {(nx - kAimLo) / (kAimHi - kAimLo) * kPlayfieldW, (ny - kAimLo) / (kAimHi - kAimLo) * kPlayfieldH};
}

class OneEuro {
public:
    OneEuro(double dt, double min_cutoff, double beta, double d_cutoff)
        : dt_(dt), min_cutoff_(min_cutoff), beta_(beta), d_cutoff_(d_cutoff) {}

    std::pair<double, double> operator()(std::pair<double, double> x) {
        if (!started_ || min_cutoff_ <= 0) {
            started_ = true;
            x_ = x;
            return x;
        }
        const double dx = (x.first - x_.first) / dt_;
        const double dy = (x.second - x_.second) / dt_;
        const double ad = alpha(d_cutoff_);
        dx_ += ad * (dx - dx_);
        dy_ += ad * (dy - dy_);
        const double a = alpha(min_cutoff_ + beta_ * std::hypot(dx_, dy_));
        x_.first += a * (x.first - x_.first);
        x_.second += a * (x.second - x_.second);
        return x_;
    }

private:
    double alpha(double cutoff) const {
        const double r = 2 * std::numbers::pi * cutoff * dt_;
        return r / (r + 1);
    }

    double dt_, min_cutoff_, beta_, d_cutoff_;
    bool started_ = false;
    std::pair<double, double> x_{0, 0};
    double dx_ = 0, dy_ = 0;
};

struct Keys {
    bool k1, k2;
};

class KeyExecutor {
public:
    KeyExecutor(int min_press_gap = 2, double press_threshold = 0.5, double rearm = 1.0, double release = 0.25)
        : min_press_gap_(min_press_gap), press_threshold_(press_threshold), rearm_(rearm), release_(release) {}

    Keys apply(double p_press, double p_down) {
        bool hold = p_down > 0.5;
        if (p_press < rearm_) armed_ = true;
        if (down_) hold = p_down > release_;
        const bool edge = prev_down_ < 0.5 && 0.5 <= p_down && !down_;
        prev_down_ = p_down;
        const bool press = ((armed_ && p_press > press_threshold_) || edge) && since_press_ >= min_press_gap_;
        if (since_press_ < std::numeric_limits<int>::max()) ++since_press_;
        if (press) {
            key_ = 1 - key_;
            down_ = true;
            since_press_ = 0;
            armed_ = false;
        } else if (!(down_ && hold)) {
            down_ = false;
        }
        pressed_ = press;
        return {down_ && key_ == 0, down_ && key_ == 1};
    }

    bool down() const { return down_; }
    int key() const { return key_; }
    bool pressed() const { return pressed_; }

private:
    bool down_ = false;
    int key_ = 1;
    bool pressed_ = false;
    int min_press_gap_;
    int since_press_ = 1000000;
    double press_threshold_;
    bool armed_ = true;
    double prev_down_ = 0.0;
    double rearm_;
    double release_;
};
