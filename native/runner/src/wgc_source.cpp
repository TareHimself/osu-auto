#include "wgc_source.h"

#include <Windows.Graphics.Capture.Interop.h>
#include <Windows.Graphics.DirectX.Direct3D11.Interop.h>
#include <dxgi.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>

#include <atomic>
#include <mutex>

#include "processor.h"
#include "window.h"

namespace wgc = winrt::Windows::Graphics::Capture;
namespace dx = winrt::Windows::Graphics::DirectX;
namespace d3d = winrt::Windows::Graphics::DirectX::Direct3D11;

struct __declspec(uuid("A9B3D012-3DF2-4EE3-B8D1-8695F457D3C1")) IDirect3DDxgiInterfaceAccess : IUnknown {
    virtual HRESULT __stdcall GetInterface(REFIID id, void** object) = 0;
};

struct WgcSource::Impl {
    static constexpr double kMinInterval = 0.008;
    static constexpr int kGeometryEvery = 60;

    HWND hwnd;
    FrameRing& ring;
    Device device = make_device();
    GpuProcessor processor{device};
    d3d::IDirect3DDevice winrt_device{nullptr};
    wgc::GraphicsCaptureItem item{nullptr};
    wgc::Direct3D11CaptureFramePool pool{nullptr};
    wgc::GraphicsCaptureSession session{nullptr};
    winrt::event_token token{};
    WindowGeometry geometry{};
    mutable std::mutex geometry_mutex;
    GrayFrame scratch{};
    int frames = 0;
    double last_t = 0.0;
    double fps_t0 = 0.0;
    std::atomic<double> fps{0.0};
    winrt::Windows::Graphics::SizeInt32 pool_size{};

    Impl(HWND hwnd_, FrameRing& ring_) : hwnd(hwnd_), ring(ring_) {
        winrt::com_ptr<IDXGIDevice> dxgi;
        winrt::check_hresult(device.device->QueryInterface(IID_PPV_ARGS(dxgi.put())));
        winrt::com_ptr<::IInspectable> inspectable;
        winrt::check_hresult(CreateDirect3D11DeviceFromDXGIDevice(dxgi.get(), inspectable.put()));
        winrt_device = inspectable.as<d3d::IDirect3DDevice>();

        auto interop = winrt::get_activation_factory<wgc::GraphicsCaptureItem, IGraphicsCaptureItemInterop>();
        winrt::check_hresult(interop->CreateForWindow(hwnd, winrt::guid_of<wgc::GraphicsCaptureItem>(), winrt::put_abi(item)));

        geometry = window_geometry(hwnd);
        pool_size = item.Size();
        pool = wgc::Direct3D11CaptureFramePool::CreateFreeThreaded(winrt_device, dx::DirectXPixelFormat::B8G8R8A8UIntNormalized, 2, pool_size);
        token = pool.FrameArrived([this](wgc::Direct3D11CaptureFramePool const& sender, winrt::Windows::Foundation::IInspectable const&) { on_frame(sender); });
        session = pool.CreateCaptureSession(item);
        session.IsCursorCaptureEnabled(false);
        try {
            session.IsBorderRequired(false);
        } catch (...) {
        }
        try {
            session.MinUpdateInterval(std::chrono::milliseconds(0));
        } catch (...) {
        }
        fps_t0 = now_seconds();
        session.StartCapture();
    }

    ~Impl() {
        pool.FrameArrived(token);
        session.Close();
        pool.Close();
    }

    void on_frame(wgc::Direct3D11CaptureFramePool const& sender) {
        auto frame = sender.TryGetNextFrame();
        if (!frame) return;
        while (auto newer = sender.TryGetNextFrame()) frame = newer;

        const double t = now_seconds();
        if (t - last_t < kMinInterval) return;
        last_t = t;

        const auto size = frame.ContentSize();
        if (size.Width != pool_size.Width || size.Height != pool_size.Height) {
            pool_size = size;
            pool.Recreate(winrt_device, dx::DirectXPixelFormat::B8G8R8A8UIntNormalized, 2, size);
        }

        winrt::com_ptr<ID3D11Texture2D> texture;
        winrt::check_hresult(frame.Surface().as<IDirect3DDxgiInterfaceAccess>()->GetInterface(IID_PPV_ARGS(texture.put())));
        processor.process(texture.get(), geometry.box, scratch);
        ring.push(t, scratch);

        if (++frames % kGeometryEvery == 0) {
            fps = kGeometryEvery / (t - fps_t0);
            fps_t0 = t;
            std::lock_guard lock(geometry_mutex);
            geometry = window_geometry(hwnd);
        }
    }
};

WgcSource::WgcSource(HWND hwnd, FrameRing& ring) : impl_(std::make_unique<Impl>(hwnd, ring)) {}
WgcSource::~WgcSource() = default;
double WgcSource::fps() const { return impl_->fps; }

WindowGeometry WgcSource::geometry() const {
    std::lock_guard lock(impl_->geometry_mutex);
    return impl_->geometry;
}
