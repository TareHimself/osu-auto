#pragma once

#include <windows.h>

#include <memory>

#include "frame_ring.h"
#include "frame_source.h"

// Windows.Graphics.Capture of one window: every frame is cropped, grayed and downsampled on the GPU, then pushed into the ring.
class WgcSource : public FrameSource {
public:
    WgcSource(HWND hwnd, FrameRing& ring);
    ~WgcSource();
    double fps() const override;
    WindowGeometry geometry() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
