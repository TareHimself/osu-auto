#pragma once

#include <windows.h>

#include <memory>

#include "frame_ring.h"
#include "frame_source.h"

// Reads the frames capture_hook.dll writes into shared memory from inside the game's Present call
class HookSource : public FrameSource {
public:
    HookSource(HWND hwnd, FrameRing& ring);
    ~HookSource() override;
    double fps() const override;
    WindowGeometry geometry() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
