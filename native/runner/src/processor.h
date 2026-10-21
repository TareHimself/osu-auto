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

// Crops a region of a BGRA texture, converts it to gray like cv2 BGRA2GRAY and area-resamples it to 160x120 like cv2 INTER_AREA.
class GpuProcessor {
public:
    explicit GpuProcessor(const Device& device);
    void process(ID3D11Texture2D* source, Crop crop, GrayFrame& out);

private:
    Device d_;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> shader_;
    Microsoft::WRL::ComPtr<ID3D11Buffer> params_;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> output_;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> output_view_;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> staging_;
};
