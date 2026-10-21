#pragma once

#include <cstdint>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

struct Npy {
    std::vector<int64_t> shape;
    std::vector<char> data;

    template <typename T>
    const T* as() const { return reinterpret_cast<const T*>(data.data()); }
};

inline Npy load_npy(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    char magic[8];
    in.read(magic, 8);
    uint32_t header_len = 0;
    if (magic[6] == 1) {
        uint16_t n = 0;
        in.read(reinterpret_cast<char*>(&n), 2);
        header_len = n;
    } else {
        in.read(reinterpret_cast<char*>(&header_len), 4);
    }
    std::string header(header_len, ' ');
    in.read(header.data(), header_len);
    if (header.find("'fortran_order': False") == std::string::npos) throw std::runtime_error(path + ": fortran order");

    Npy npy;
    const auto open = header.find('(', header.find("'shape'"));
    const auto close = header.find(')', open);
    const std::string dims = header.substr(open + 1, close - open - 1);
    const std::regex digits("\\d+");
    for (std::sregex_iterator it(dims.begin(), dims.end(), digits), end; it != end; ++it) {
        npy.shape.push_back(std::stoll(it->str()));
    }
    const auto begin = in.tellg();
    in.seekg(0, std::ios::end);
    npy.data.resize(static_cast<size_t>(in.tellg() - begin));
    in.seekg(begin);
    in.read(npy.data.data(), static_cast<std::streamsize>(npy.data.size()));
    return npy;
}
