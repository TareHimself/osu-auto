#include <windows.h>

#include <d3d11.h>
#include <dxgi.h>
#include <dxgi1_2.h>
#include <wrl/client.h>

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <deque>
#include <memory>
#include <mutex>
#include <string>

#include "hook_shm.h"
#include "processor.h"
#include "window.h"

using Microsoft::WRL::ComPtr;

namespace {

constexpr int kPresentSlot = 8;
constexpr int kResizeSlot = 13;
constexpr int kPresent1Slot = 22;
constexpr double kMinIntervalS = 0.002;
constexpr size_t kMaxPending = 2;

using PresentFn = HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain*, UINT, UINT);
using Present1Fn = HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain1*, UINT, UINT, const DXGI_PRESENT_PARAMETERS*);
using ResizeFn = HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain*, UINT, UINT, UINT, DXGI_FORMAT, UINT);

void** g_vtable = nullptr;
PresentFn g_present = nullptr;
Present1Fn g_present1 = nullptr;
ResizeFn g_resize = nullptr;
void* g_hooked_present = nullptr;
void* g_hooked_present1 = nullptr;
void* g_hooked_resize = nullptr;

std::atomic<bool> g_disabled{false};
std::atomic<int> g_inflight{0};
thread_local bool t_busy = false;

void log(const char* format, ...) {
    char path[MAX_PATH];
    const DWORD n = GetTempPathA(MAX_PATH, path);
    std::snprintf(path + n, MAX_PATH - n, "osu-auto-hook.log");
    FILE* f = nullptr;
    if (fopen_s(&f, path, "a") != 0 || !f) return;
    va_list args;
    va_start(args, format);
    std::vfprintf(f, format, args);
    va_end(args);
    std::fputc('\n', f);
    std::fclose(f);
}

struct Capture {
    std::mutex mutex;
    bool failed = false;
    ComPtr<IDXGISwapChain> chain;
    Device device;
    std::unique_ptr<GpuProcessor> processor;
    ComPtr<ID3D11Texture2D> copy;
    DXGI_FORMAT view_format = DXGI_FORMAT_UNKNOWN;
    UINT width = 0, height = 0;
    Crop box{};
    std::deque<int> pending;
    double last_capture = 0;
    HANDLE mapping = nullptr;
    HANDLE frame_event = nullptr;
    HookShm* shm = nullptr;

    void reset_gpu() {
        pending.clear();
        copy.Reset();
        processor.reset();
        device = Device{};
        chain.Reset();
        width = height = 0;
    }

    bool open_shared_memory() {
        if (shm) return true;
        mapping = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, sizeof(HookShm), kHookShmName);
        if (!mapping) return false;
        shm = static_cast<HookShm*>(MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(HookShm)));
        frame_event = CreateEventW(nullptr, FALSE, FALSE, kHookEventName);
        if (!shm || !frame_event) return false;
        shm->magic = kHookMagic;
        shm->version = kHookVersion;
        shm->width = kFrameW;
        shm->height = kFrameH;
        return true;
    }

    bool init(IDXGISwapChain* sc) {
        DXGI_SWAP_CHAIN_DESC desc;
        if (FAILED(sc->GetDesc(&desc))) return false;
        wchar_t title[128]{};
        GetWindowTextW(desc.OutputWindow, title, 128);
        if (std::wstring(title).rfind(L"osu", 0) != 0) return false;

        ComPtr<ID3D11Device> dev;
        if (FAILED(sc->GetDevice(IID_PPV_ARGS(&dev)))) {
            log("swapchain is not D3D11, hook disabled");
            failed = true;
            return false;
        }
        device.device = dev;
        dev->GetImmediateContext(&device.context);
        chain = sc;
        if (!open_shared_memory()) {
            log("shared memory failed: %lu", GetLastError());
            failed = true;
            return false;
        }
        try {
            processor = std::make_unique<GpuProcessor>(device);
        } catch (const std::exception& e) {
            log("processor failed: %s", e.what());
            failed = true;
            return false;
        }
        log("hooked swapchain %p of %ls", static_cast<void*>(sc), title);
        return true;
    }

    // Recreates the size dependent resources, returns false if this backbuffer cannot be used
    bool prepare(ID3D11Texture2D* backbuffer) {
        D3D11_TEXTURE2D_DESC bd;
        backbuffer->GetDesc(&bd);
        if (bd.SampleDesc.Count > 1) {
            log("multisampled backbuffer, hook disabled");
            failed = true;
            return false;
        }
        DXGI_FORMAT typeless;
        switch (bd.Format) {
            case DXGI_FORMAT_B8G8R8A8_UNORM:
            case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
            case DXGI_FORMAT_B8G8R8A8_TYPELESS:
                typeless = DXGI_FORMAT_B8G8R8A8_TYPELESS;
                view_format = DXGI_FORMAT_B8G8R8A8_UNORM;
                break;
            case DXGI_FORMAT_R8G8B8A8_UNORM:
            case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
            case DXGI_FORMAT_R8G8B8A8_TYPELESS:
                typeless = DXGI_FORMAT_R8G8B8A8_TYPELESS;
                view_format = DXGI_FORMAT_R8G8B8A8_UNORM;
                break;
            default:
                log("unsupported backbuffer format %d, hook disabled", static_cast<int>(bd.Format));
                failed = true;
                return false;
        }
        width = bd.Width;
        height = bd.Height;
        box = capture_box(static_cast<int>(width), static_cast<int>(height), 0, 0);
        D3D11_TEXTURE2D_DESC cd{};
        cd.Width = static_cast<UINT>(box.w);
        cd.Height = static_cast<UINT>(box.h);
        cd.MipLevels = 1;
        cd.ArraySize = 1;
        cd.Format = typeless;
        cd.SampleDesc.Count = 1;
        cd.Usage = D3D11_USAGE_DEFAULT;
        cd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        copy.Reset();
        pending.clear();
        if (FAILED(device.device->CreateTexture2D(&cd, nullptr, &copy))) {
            log("copy texture failed");
            failed = true;
            return false;
        }
        log("capturing %ux%u, box %d,%d %dx%d", width, height, box.x, box.y, box.w, box.h);
        return true;
    }

    void publish(const GrayFrame& frame, double t) {
        LARGE_INTEGER c;
        QueryPerformanceCounter(&c);
        (void)t;
        shm->seq = shm->seq + 1;
        MemoryBarrier();
        std::memcpy(shm->frame, frame.data(), frame.size());
        shm->qpc = c.QuadPart;
        MemoryBarrier();
        shm->seq = shm->seq + 1;
        SetEvent(frame_event);
    }

    void on_present(IDXGISwapChain* sc) {
        std::lock_guard lock(mutex);
        if (failed) return;
        if (!processor && !init(sc)) return;
        if (chain.Get() != sc) return;

        const double now = now_seconds();
        ComPtr<ID3D11Texture2D> backbuffer;
        if (FAILED(sc->GetBuffer(0, IID_PPV_ARGS(&backbuffer)))) return;
        D3D11_TEXTURE2D_DESC bd;
        backbuffer->GetDesc(&bd);
        if (!copy || bd.Width != width || bd.Height != height) {
            if (!prepare(backbuffer.Get())) return;
        }

        GrayFrame frame;
        while (!pending.empty() && processor->read(pending.front(), frame, false)) {
            pending.pop_front();
            publish(frame, now);
        }
        if (pending.size() >= kMaxPending || now - last_capture < kMinIntervalS) return;
        last_capture = now;

        const D3D11_BOX region{static_cast<UINT>(box.x), static_cast<UINT>(box.y), 0, static_cast<UINT>(box.x + box.w), static_cast<UINT>(box.y + box.h), 1};
        device.context->CopySubresourceRegion(copy.Get(), 0, 0, 0, 0, backbuffer.Get(), 0, &region);
        pending.push_back(processor->dispatch(copy.Get(), Crop{0, 0, box.w, box.h}, view_format));
    }

    void on_resize(IDXGISwapChain* sc) {
        std::lock_guard lock(mutex);
        if (chain.Get() == sc) {
            copy.Reset();
            pending.clear();
        }
    }
};

Capture g_capture;

struct InFlight {
    InFlight() { ++g_inflight; }
    ~InFlight() { --g_inflight; }
};

HRESULT STDMETHODCALLTYPE hooked_present(IDXGISwapChain* sc, UINT sync, UINT flags) {
    InFlight in_flight;
    if (!g_disabled && !t_busy && !(flags & DXGI_PRESENT_TEST)) {
        t_busy = true;
        try {
            g_capture.on_present(sc);
        } catch (const std::exception& e) {
            log("present failed: %s", e.what());
            g_capture.failed = true;
        }
        t_busy = false;
    }
    return g_present(sc, sync, flags);
}

HRESULT STDMETHODCALLTYPE hooked_present1(IDXGISwapChain1* sc, UINT sync, UINT flags, const DXGI_PRESENT_PARAMETERS* parameters) {
    InFlight in_flight;
    if (!g_disabled && !t_busy && !(flags & DXGI_PRESENT_TEST)) {
        t_busy = true;
        try {
            g_capture.on_present(sc);
        } catch (const std::exception& e) {
            log("present1 failed: %s", e.what());
            g_capture.failed = true;
        }
        t_busy = false;
    }
    return g_present1(sc, sync, flags, parameters);
}

HRESULT STDMETHODCALLTYPE hooked_resize(IDXGISwapChain* sc, UINT count, UINT width, UINT height, DXGI_FORMAT format, UINT flags) {
    InFlight in_flight;
    if (!g_disabled) g_capture.on_resize(sc);
    return g_resize(sc, count, width, height, format, flags);
}

void patch_slot(int slot, void* replacement) {
    DWORD old;
    VirtualProtect(&g_vtable[slot], sizeof(void*), PAGE_READWRITE, &old);
    g_vtable[slot] = replacement;
    VirtualProtect(&g_vtable[slot], sizeof(void*), old, &old);
}

DWORD WINAPI install(LPVOID) {
    WNDCLASSW wc{};
    wc.lpfnWndProc = DefWindowProcW;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"osu-auto-hook-probe";
    RegisterClassW(&wc);
    HWND window = CreateWindowExW(0, wc.lpszClassName, L"", WS_OVERLAPPED, 0, 0, 64, 64, nullptr, nullptr, wc.hInstance, nullptr);

    DXGI_SWAP_CHAIN_DESC sd{};
    sd.BufferCount = 1;
    sd.BufferDesc.Width = 64;
    sd.BufferDesc.Height = 64;
    sd.BufferDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = window;
    sd.SampleDesc.Count = 1;
    sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

    ComPtr<IDXGISwapChain> chain;
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    const HRESULT hr = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &sd, &chain, &device, nullptr, &context);
    if (FAILED(hr)) {
        log("probe swapchain failed: 0x%08lx", static_cast<unsigned long>(hr));
        DestroyWindow(window);
        return 1;
    }
    g_vtable = *reinterpret_cast<void***>(chain.Get());
    g_present = reinterpret_cast<PresentFn>(g_vtable[kPresentSlot]);
    g_present1 = reinterpret_cast<Present1Fn>(g_vtable[kPresent1Slot]);
    g_resize = reinterpret_cast<ResizeFn>(g_vtable[kResizeSlot]);
    g_hooked_present = reinterpret_cast<void*>(&hooked_present);
    g_hooked_present1 = reinterpret_cast<void*>(&hooked_present1);
    g_hooked_resize = reinterpret_cast<void*>(&hooked_resize);
    patch_slot(kPresentSlot, g_hooked_present);
    patch_slot(kPresent1Slot, g_hooked_present1);
    patch_slot(kResizeSlot, g_hooked_resize);
    log("installed, vtable %p", static_cast<void*>(g_vtable));

    chain.Reset();
    context.Reset();
    device.Reset();
    DestroyWindow(window);
    UnregisterClassW(wc.lpszClassName, wc.hInstance);
    return 0;
}

void restore_slot(int slot, void* ours, void* original) {
    if (g_vtable && g_vtable[slot] == ours) patch_slot(slot, original);
}

}  // namespace

extern "C" __declspec(dllexport) DWORD WINAPI HookEject(LPVOID) {
    g_disabled = true;
    restore_slot(kPresentSlot, g_hooked_present, reinterpret_cast<void*>(g_present));
    restore_slot(kPresent1Slot, g_hooked_present1, reinterpret_cast<void*>(g_present1));
    restore_slot(kResizeSlot, g_hooked_resize, reinterpret_cast<void*>(g_resize));
    for (int i = 0; i < 200 && g_inflight > 0; ++i) Sleep(5);
    Sleep(100);
    {
        std::lock_guard lock(g_capture.mutex);
        g_capture.reset_gpu();
    }
    log("ejected");
    return 0;
}

BOOL WINAPI DllMain(HINSTANCE module, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        DisableThreadLibraryCalls(module);
        if (HANDLE thread = CreateThread(nullptr, 0, install, nullptr, 0, nullptr)) CloseHandle(thread);
    }
    return TRUE;
}
