#include "processor.h"

#include <d3dcompiler.h>

#include <cstring>
#include <stdexcept>
#include <string>

using Microsoft::WRL::ComPtr;

namespace {

constexpr char kShader[] = R"(
cbuffer Params : register(b0) {
    int2 origin;
    int2 size;
    int2 bounds;
    int2 pad;
};
Texture2D<float4> src : register(t0);
RWTexture2D<unorm float> dst : register(u0);

uint gray(int2 p) {
    float3 v = src.Load(int3(clamp(p, int2(0, 0), bounds - 1), 0)).rgb;
    uint3 c = (uint3)round(v * 255.0);
    return (c.r * 4899u + c.g * 9617u + c.b * 1868u + 8192u) >> 14;
}

[numthreads(16, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint w, h;
    dst.GetDimensions(w, h);
    if (id.x >= w || id.y >= h) return;
    float2 scale = float2(size) / float2(w, h);
    float2 lo = float2(id.xy) * scale;
    float2 hi = lo + scale;
    int2 i0 = (int2)floor(lo);
    int2 i1 = (int2)ceil(hi);
    float sum = 0;
    for (int y = i0.y; y < i1.y; ++y) {
        float wy = min(hi.y, y + 1.0) - max(lo.y, (float)y);
        for (int x = i0.x; x < i1.x; ++x) {
            float wx = min(hi.x, x + 1.0) - max(lo.x, (float)x);
            sum += wx * wy * gray(origin + int2(x, y));
        }
    }
    dst[id.xy] = floor(sum / (scale.x * scale.y) + 0.5) / 255.0;
}
)";

void check(HRESULT hr, const char* what) {
    if (FAILED(hr)) throw std::runtime_error(std::string(what) + " failed: 0x" + std::to_string(static_cast<unsigned>(hr)));
}

struct Params {
    int origin[2];
    int size[2];
    int bounds[2];
    int pad[2];
};

}  // namespace

Device make_device() {
    Device d;
    check(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, D3D11_CREATE_DEVICE_BGRA_SUPPORT, nullptr, 0, D3D11_SDK_VERSION, &d.device, nullptr, &d.context),
          "D3D11CreateDevice");
    return d;
}

GpuProcessor::GpuProcessor(const Device& device) : d_(device) {
    ComPtr<ID3DBlob> code, errors;
    const HRESULT hr = D3DCompile(kShader, sizeof(kShader) - 1, "gray_area", nullptr, nullptr, "main", "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &errors);
    if (FAILED(hr)) throw std::runtime_error(std::string("shader: ") + (errors ? static_cast<const char*>(errors->GetBufferPointer()) : "unknown"));
    check(d_.device->CreateComputeShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr, &shader_), "CreateComputeShader");

    D3D11_BUFFER_DESC bd{};
    bd.ByteWidth = sizeof(Params);
    bd.Usage = D3D11_USAGE_DEFAULT;
    bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    check(d_.device->CreateBuffer(&bd, nullptr, &params_), "CreateBuffer");

    D3D11_TEXTURE2D_DESC td{};
    td.Width = kFrameW;
    td.Height = kFrameH;
    td.MipLevels = 1;
    td.ArraySize = 1;
    td.Format = DXGI_FORMAT_R8_UNORM;
    td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    check(d_.device->CreateTexture2D(&td, nullptr, &output_), "CreateTexture2D output");
    check(d_.device->CreateUnorderedAccessView(output_.Get(), nullptr, &output_view_), "CreateUnorderedAccessView");

    td.Usage = D3D11_USAGE_STAGING;
    td.BindFlags = 0;
    td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    check(d_.device->CreateTexture2D(&td, nullptr, &staging_), "CreateTexture2D staging");
}

void GpuProcessor::process(ID3D11Texture2D* source, Crop crop, GrayFrame& out) {
    D3D11_TEXTURE2D_DESC sd;
    source->GetDesc(&sd);
    ComPtr<ID3D11ShaderResourceView> view;
    check(d_.device->CreateShaderResourceView(source, nullptr, &view), "CreateShaderResourceView");

    const Params p{{crop.x, crop.y}, {crop.w, crop.h}, {static_cast<int>(sd.Width), static_cast<int>(sd.Height)}, {0, 0}};
    d_.context->UpdateSubresource(params_.Get(), 0, nullptr, &p, 0, 0);

    ID3D11Buffer* cb = params_.Get();
    ID3D11ShaderResourceView* srv = view.Get();
    ID3D11UnorderedAccessView* uav = output_view_.Get();
    d_.context->CSSetShader(shader_.Get(), nullptr, 0);
    d_.context->CSSetConstantBuffers(0, 1, &cb);
    d_.context->CSSetShaderResources(0, 1, &srv);
    d_.context->CSSetUnorderedAccessViews(0, 1, &uav, nullptr);
    d_.context->Dispatch((kFrameW + 15) / 16, (kFrameH + 7) / 8, 1);

    ID3D11ShaderResourceView* no_srv = nullptr;
    ID3D11UnorderedAccessView* no_uav = nullptr;
    d_.context->CSSetShaderResources(0, 1, &no_srv);
    d_.context->CSSetUnorderedAccessViews(0, 1, &no_uav, nullptr);

    d_.context->CopyResource(staging_.Get(), output_.Get());
    D3D11_MAPPED_SUBRESOURCE m;
    check(d_.context->Map(staging_.Get(), 0, D3D11_MAP_READ, 0, &m), "Map");
    for (int row = 0; row < kFrameH; ++row) {
        std::memcpy(out.data() + row * kFrameW, static_cast<const uint8_t*>(m.pData) + row * m.RowPitch, kFrameW);
    }
    d_.context->Unmap(staging_.Get(), 0);
}
