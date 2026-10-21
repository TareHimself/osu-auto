#pragma once

#include "window.h"

class FrameSource {
public:
    virtual ~FrameSource() = default;
    virtual double fps() const = 0;
    virtual WindowGeometry geometry() const = 0;
};
