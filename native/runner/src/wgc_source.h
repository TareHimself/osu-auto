#pragma once

#include <windows.h>

#include <memory>

#include "frame_ring.h"
#include "window.h"

// Windows.Graphics.Capture of one window: every frame is cropped, grayed and downsampled on the GPU, then pushed into the ring.
class WgcSource {
public:
    WgcSource(HWND hwnd, FrameRing& ring);
    ~WgcSource();
    double fps() const;
    WindowGeometry geometry() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
