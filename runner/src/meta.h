#pragma once

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct ModelMeta {
    int stack = 4;
    std::vector<int> offsets;
    double latency_max_ms = 0.0;
    int frame_h = 120;
    int frame_w = 160;

    // Frame k of the stack is the one nearest now - steps[k] * step, oldest first
    std::vector<int> steps() const {
        std::vector<int> out;
        if (offsets.empty()) {
            for (int k = stack - 1; k >= 0; --k) out.push_back(k);
        } else {
            out.assign(offsets.rbegin(), offsets.rend());
        }
        return out;
    }
};

// Reads the flat json that export_onnx.py writes next to the model
inline ModelMeta load_meta(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::stringstream buffer;
    buffer << in.rdbuf();
    const std::string text = buffer.str();

    auto value_at = [&](const std::string& key) {
        const auto k = text.find("\"" + key + "\"");
        if (k == std::string::npos) throw std::runtime_error(path + ": missing " + key);
        return text.c_str() + text.find(':', k) + 1;
    };
    auto numbers_in = [](const char* p) {
        std::vector<double> out;
        while (*p && *p != '[') ++p;
        if (!*p) return out;
        ++p;
        while (*p && *p != ']') {
            char* end = nullptr;
            const double v = std::strtod(p, &end);
            if (end == p) {
                ++p;
                continue;
            }
            out.push_back(v);
            p = end;
        }
        return out;
    };

    ModelMeta m;
    m.stack = static_cast<int>(std::strtol(value_at("stack"), nullptr, 10));
    m.latency_max_ms = std::strtod(value_at("latency_max_ms"), nullptr);
    const char* offsets = value_at("offsets");
    while (*offsets == ' ' || *offsets == '\n') ++offsets;
    if (*offsets == '[') {
        for (double v : numbers_in(offsets)) m.offsets.push_back(static_cast<int>(v));
    }
    const auto hw = numbers_in(value_at("frame_hw"));
    if (hw.size() == 2) {
        m.frame_h = static_cast<int>(hw[0]);
        m.frame_w = static_cast<int>(hw[1]);
    }
    return m;
}
