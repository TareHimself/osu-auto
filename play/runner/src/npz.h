#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace npz {

using Bytes = std::vector<char>;

inline uint32_t crc32(const Bytes& data) {
    static const auto table = [] {
        std::array<uint32_t, 256> t{};
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            t[i] = c;
        }
        return t;
    }();
    uint32_t c = 0xFFFFFFFFu;
    for (char b : data) c = table[(c ^ static_cast<uint8_t>(b)) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

inline Bytes npy(const std::string& descr, const std::string& shape, const Bytes& payload) {
    std::string header = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + shape + ", }";
    while ((10 + header.size() + 1) % 64 != 0) header += ' ';
    header += '\n';
    Bytes out = {'\x93', 'N', 'U', 'M', 'P', 'Y', '\x01', '\x00'};
    const uint16_t n = static_cast<uint16_t>(header.size());
    out.insert(out.end(), reinterpret_cast<const char*>(&n), reinterpret_cast<const char*>(&n) + 2);
    out.insert(out.end(), header.begin(), header.end());
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
}

inline Bytes matrix_f64(const std::vector<double>& values, size_t rows, size_t cols) {
    Bytes payload(values.size() * sizeof(double));
    std::memcpy(payload.data(), values.data(), payload.size());
    return npy("<f8", "(" + std::to_string(rows) + ", " + std::to_string(cols) + ")", payload);
}

// Fixed-width UTF-32 strings, numpy's <U<width>
inline Bytes strings(const std::vector<std::string>& values, size_t width) {
    Bytes payload(values.size() * width * 4, 0);
    for (size_t i = 0; i < values.size(); ++i) {
        for (size_t j = 0; j < values[i].size() && j < width; ++j) payload[(i * width + j) * 4] = values[i][j];
    }
    return npy("<U" + std::to_string(width), "(" + std::to_string(values.size()) + ",)", payload);
}

template <typename T>
void put(Bytes& out, T v) {
    out.insert(out.end(), reinterpret_cast<const char*>(&v), reinterpret_cast<const char*>(&v) + sizeof(T));
}

// An uncompressed zip, which is all numpy.load needs
inline void write(const std::string& path, const std::vector<std::pair<std::string, Bytes>>& entries) {
    Bytes file, central;
    for (const auto& [name, data] : entries) {
        const uint32_t offset = static_cast<uint32_t>(file.size());
        const uint32_t crc = crc32(data), size = static_cast<uint32_t>(data.size());
        put<uint32_t>(file, 0x04034b50);
        put<uint16_t>(file, 20);
        put<uint16_t>(file, 0);
        put<uint16_t>(file, 0);
        put<uint16_t>(file, 0);
        put<uint16_t>(file, 0x21);
        put<uint32_t>(file, crc);
        put<uint32_t>(file, size);
        put<uint32_t>(file, size);
        put<uint16_t>(file, static_cast<uint16_t>(name.size()));
        put<uint16_t>(file, 0);
        file.insert(file.end(), name.begin(), name.end());
        file.insert(file.end(), data.begin(), data.end());

        put<uint32_t>(central, 0x02014b50);
        put<uint16_t>(central, 20);
        put<uint16_t>(central, 20);
        put<uint16_t>(central, 0);
        put<uint16_t>(central, 0);
        put<uint16_t>(central, 0);
        put<uint16_t>(central, 0x21);
        put<uint32_t>(central, crc);
        put<uint32_t>(central, size);
        put<uint32_t>(central, size);
        put<uint16_t>(central, static_cast<uint16_t>(name.size()));
        put<uint16_t>(central, 0);
        put<uint16_t>(central, 0);
        put<uint16_t>(central, 0);
        put<uint16_t>(central, 0);
        put<uint32_t>(central, 0);
        put<uint32_t>(central, offset);
        central.insert(central.end(), name.begin(), name.end());
    }
    const uint32_t central_offset = static_cast<uint32_t>(file.size());
    file.insert(file.end(), central.begin(), central.end());
    put<uint32_t>(file, 0x06054b50);
    put<uint16_t>(file, 0);
    put<uint16_t>(file, 0);
    put<uint16_t>(file, static_cast<uint16_t>(entries.size()));
    put<uint16_t>(file, static_cast<uint16_t>(entries.size()));
    put<uint32_t>(file, static_cast<uint32_t>(central.size()));
    put<uint32_t>(file, central_offset);
    put<uint16_t>(file, 0);
    std::ofstream out(path, std::ios::binary);
    out.write(file.data(), static_cast<std::streamsize>(file.size()));
}

}  // namespace npz
