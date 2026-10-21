#pragma once

#include <d3d11.h>
#include <wrl/client.h>

#include "frame_ring.h"

struct Crop {
    int x, y, w, h;
};

struct Device {
    Microsoft::WRL::ComPtr<ID3D11Device> device;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> context;
};

Device make_device();

// Crops a region of a BGRA or RGBA texture, converts it to gray like cv2 BGRA2GRAY and area-resamples it to 160x120 like cv2 INTER_AREA.
// `view_format` is for typeless sources and is the format the shader reads them as.
class GpuProcessor {
public:
    static constexpr int kSlots = 3;

    explicit GpuProcessor(const Device& device);

    void process(ID3D11Texture2D* source, Crop crop, GrayFrame& out, DXGI_FORMAT view_format = DXGI_FORMAT_UNKNOWN);

    // Queues the work into a staging slot and returns immediately, so the render thread never waits on the GPU
    int dispatch(ID3D11Texture2D* source, Crop crop, DXGI_FORMAT view_format = DXGI_FORMAT_UNKNOWN);
    // False when the GPU is not done with that slot yet and `wait` is off
    bool read(int slot, GrayFrame& out, bool wait);

private:
    Device d_;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> shader_;
    Microsoft::WRL::ComPtr<ID3D11Buffer> params_;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> output_;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> output_view_;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> staging_[kSlots];
    int next_slot_ = 0;
};
