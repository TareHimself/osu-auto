#include "hook_source.h"

#include <atomic>
#include <stdexcept>
#include <thread>

#include "hook_shm.h"

struct HookSource::Impl {
    HWND hwnd;
    FrameRing& ring;
    HANDLE mapping = nullptr;
    HANDLE frame_event = nullptr;
    const HookShm* shm = nullptr;
    std::atomic<bool> closing{false};
    std::atomic<double> fps{0.0};
    std::thread reader;

    Impl(HWND hwnd_, FrameRing& ring_) : hwnd(hwnd_), ring(ring_) {
        mapping = OpenFileMappingW(FILE_MAP_READ, FALSE, kHookShmName);
        if (!mapping) throw std::runtime_error("the capture hook is not loaded (run: runner hook inject)");
        shm = static_cast<const HookShm*>(MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, sizeof(HookShm)));
        frame_event = OpenEventW(SYNCHRONIZE, FALSE, kHookEventName);
        if (!shm || !frame_event || shm->magic != kHookMagic || shm->version != kHookVersion) throw std::runtime_error("the capture hook shared memory is not valid");
        reader = std::thread([this] { run(); });
    }

    ~Impl() {
        closing = true;
        if (reader.joinable()) reader.join();
        if (shm) UnmapViewOfFile(shm);
        if (frame_event) CloseHandle(frame_event);
        if (mapping) CloseHandle(mapping);
    }

    void run() {
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
        uint64_t last_seq = shm->seq & ~1ull;
        int frames = 0;
        double fps_t0 = now_seconds();
        GrayFrame frame;
        while (!closing) {
            if (WaitForSingleObject(frame_event, 100) != WAIT_OBJECT_0) continue;
            for (int attempt = 0; attempt < 5; ++attempt) {
                const uint64_t before = shm->seq;
                if (before & 1) continue;
                std::memcpy(frame.data(), shm->frame, frame.size());
                const int64_t qpc = shm->qpc;
                MemoryBarrier();
                if (shm->seq != before) continue;
                if (before != last_seq) {
                    last_seq = before;
                    ring.push(qpc_to_seconds(qpc), frame);
                    if (++frames % 60 == 0) {
                        const double now = now_seconds();
                        fps = 60 / (now - fps_t0);
                        fps_t0 = now;
                    }
                }
                break;
            }
        }
    }
};

HookSource::HookSource(HWND hwnd, FrameRing& ring) : impl_(std::make_unique<Impl>(hwnd, ring)) {}
HookSource::~HookSource() = default;
double HookSource::fps() const { return impl_->fps; }
WindowGeometry HookSource::geometry() const { return window_geometry(impl_->hwnd); }
