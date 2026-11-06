#include <winsock2.h>
#include <ws2tcpip.h>

#include "overlay.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace {

constexpr double kSendHz = 60.0;
constexpr int kIdleResendMs = 250;

std::string base64(const uint8_t* data, size_t n) {
    static constexpr char table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((n + 2) / 3 * 4);
    for (size_t i = 0; i < n; i += 3) {
        const uint32_t a = data[i], b = i + 1 < n ? data[i + 1] : 0, c = i + 2 < n ? data[i + 2] : 0;
        const uint32_t v = (a << 16) | (b << 8) | c;
        out += table[(v >> 18) & 63];
        out += table[(v >> 12) & 63];
        out += i + 1 < n ? table[(v >> 6) & 63] : '=';
        out += i + 2 < n ? table[v & 63] : '=';
    }
    return out;
}

bool send_all(SOCKET s, const std::string& data) {
    size_t sent = 0;
    while (sent < data.size()) {
        const int n = send(s, data.data() + sent, static_cast<int>(data.size() - sent), 0);
        if (n <= 0) return false;
        sent += static_cast<size_t>(n);
    }
    return true;
}

const char* content_type(const std::filesystem::path& p) {
    const auto ext = p.extension().string();
    if (ext == ".html") return "text/html; charset=utf-8";
    if (ext == ".js") return "text/javascript";
    if (ext == ".css") return "text/css";
    if (ext == ".svg") return "image/svg+xml";
    if (ext == ".png") return "image/png";
    return "application/octet-stream";
}

bool safe_name(const std::string& name) {
    return !name.empty() && name.find("..") == std::string::npos &&
           std::all_of(name.begin(), name.end(), [](char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '.' || c == '-' || c == '_'; });
}

std::string encode(const OverlaySnapshot& s) {
    std::string out = "data: {";
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "\"active\":%s,\"focused\":%s,\"x\":%.5f,\"y\":%.5f,\"p_press\":%.4f,\"p_down\":%.4f,\"k1\":%d,\"k2\":%d,\"key_names\":\"%c%c\",\"presses\":%lld,\"k1_presses\":%lld,\"k2_presses\":%lld,"
                  "\"frame_age_ms\":%.2f,\"infer_ms\":%.2f,\"latency_ms\":%.2f,\"capture_fps\":%.0f,\"fw\":%d,\"fh\":%d,\"aw\":%d,\"ah\":%d,",
                  s.active ? "true" : "false", s.focused ? "true" : "false", s.box_x, s.box_y, s.p_press, s.p_down, s.k1 ? 1 : 0, s.k2 ? 1 : 0, s.key_names[0], s.key_names[1], s.presses, s.k1_presses, s.k2_presses,
                  s.frame_age_ms, s.infer_ms, s.latency_ms, s.capture_fps, kFrameW, kFrameH, s.attention_w, s.attention_h);
    out += buf;
    out += "\"frame\":\"" + base64(s.frame.data(), s.frame.size()) + "\",";
    out += "\"attention\":\"" + base64(reinterpret_cast<const uint8_t*>(s.attention.data()), s.attention.size() * sizeof(float)) + "\"}\n\n";
    return out;
}

}  // namespace

OverlayServer::OverlayServer(int port, std::filesystem::path dir) : dir_(std::move(dir)) {
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) throw std::runtime_error("WSAStartup failed");
    const SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<u_short>(port));
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (s == INVALID_SOCKET || bind(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 || listen(s, 8) != 0) {
        if (s != INVALID_SOCKET) closesocket(s);
        WSACleanup();
        throw std::runtime_error("overlay: cannot listen on 127.0.0.1:" + std::to_string(port));
    }
    listener_ = s;
    thread_ = std::thread([this] { serve(); });
}

OverlayServer::~OverlayServer() {
    closing_ = true;
    if (thread_.joinable()) thread_.join();
    for (auto c : clients_) closesocket(c);
    closesocket(listener_);
    WSACleanup();
}

void OverlayServer::publish(const OverlaySnapshot& snapshot) {
    std::lock_guard lock(mutex_);
    snapshot_ = snapshot;
    ++version_;
}

void OverlayServer::set_active(bool active) {
    std::lock_guard lock(mutex_);
    snapshot_.active = active;
    ++version_;
}

void OverlayServer::serve() {
    const auto period = std::chrono::duration<double>(1.0 / kSendHz);
    auto next = std::chrono::steady_clock::now();
    auto last_send = next;
    uint64_t sent_version = 0;
    while (!closing_) {
        fd_set read;
        FD_ZERO(&read);
        FD_SET(static_cast<SOCKET>(listener_), &read);
        const auto wait = std::max(std::chrono::duration<double>(0), std::chrono::duration<double>(next - std::chrono::steady_clock::now()));
        timeval tv{0, static_cast<long>(wait.count() * 1e6)};
        if (select(0, &read, nullptr, nullptr, &tv) > 0) {
            const SOCKET c = accept(static_cast<SOCKET>(listener_), nullptr, nullptr);
            if (c != INVALID_SOCKET) handle(c);
        }
        const auto now = std::chrono::steady_clock::now();
        if (now < next) continue;
        next += std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);
        if (next < now - period) next = now;

        std::string message;
        {
            std::lock_guard lock(mutex_);
            const bool fresh = version_ != sent_version;
            if (clients_.empty() || (!fresh && now - last_send < std::chrono::milliseconds(kIdleResendMs))) continue;
            sent_version = version_;
            message = encode(snapshot_);
        }
        last_send = now;
        std::erase_if(clients_, [&](uintptr_t c) {
            if (send_all(static_cast<SOCKET>(c), message)) return false;
            closesocket(static_cast<SOCKET>(c));
            return true;
        });
    }
}

void OverlayServer::handle(uintptr_t client) {
    const SOCKET c = static_cast<SOCKET>(client);
    const DWORD timeout_ms = 500;
    setsockopt(c, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&timeout_ms), sizeof(timeout_ms));
    setsockopt(c, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&timeout_ms), sizeof(timeout_ms));
    const BOOL nodelay = TRUE;
    setsockopt(c, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&nodelay), sizeof(nodelay));

    std::string request;
    char buf[2048];
    while (request.find("\r\n\r\n") == std::string::npos && request.size() < 16384) {
        const int n = recv(c, buf, sizeof(buf), 0);
        if (n <= 0) break;
        request.append(buf, static_cast<size_t>(n));
    }
    std::string path;
    if (request.rfind("GET ", 0) == 0) {
        const auto end = request.find(' ', 4);
        if (end != std::string::npos) path = request.substr(4, end - 4);
    }
    if (const auto q = path.find('?'); q != std::string::npos) path.resize(q);

    if (path == "/events") {
        if (send_all(c, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\nConnection: keep-alive\r\n\r\n")) {
            clients_.push_back(client);
            return;
        }
        closesocket(c);
        return;
    }

    const std::string name = path == "/" ? "index.html" : path.substr(path.empty() ? 0 : 1);
    std::string body;
    std::string status = "404 Not Found";
    const auto file = dir_ / name;
    if (safe_name(name)) {
        std::ifstream in(file, std::ios::binary);
        if (in) {
            body.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
            status = "200 OK";
        }
    }
    const std::string head = "HTTP/1.1 " + status + "\r\nContent-Type: " + (status[0] == '2' ? content_type(file) : "text/plain") +
                             "\r\nCache-Control: no-cache\r\nContent-Length: " + std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n";
    send_all(c, head + body);
    shutdown(c, SD_SEND);
    closesocket(c);
}
