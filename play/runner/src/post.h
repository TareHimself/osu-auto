#pragma once

#include <cmath>
#include <limits>
#include <utility>

constexpr double kPlayfieldW = 512.0;
constexpr double kPlayfieldH = 384.0;
constexpr double kAimLo = 0.25;
constexpr double kAimHi = 0.75;

inline std::pair<double, double> decode_to_playfield(double nx, double ny) {
    return {(nx - kAimLo) / (kAimHi - kAimLo) * kPlayfieldW, (ny - kAimLo) / (kAimHi - kAimLo) * kPlayfieldH};
}

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
