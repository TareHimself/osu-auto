#pragma once

#include <windows.h>

#include <optional>
#include <string>

#include "processor.h"

struct WindowGeometry {
    int client_w, client_h;
    int client_x, client_y;  // client origin on screen
    Crop box;                // 4:3 capture box inside the captured frame
};

std::optional<HWND> find_window(const std::wstring& title_prefix);
WindowGeometry window_geometry(HWND hwnd);
Crop capture_box(int client_w, int client_h, int offset_x, int offset_y);

// Playfield units (512x384) to screen pixels, from the client size like PlayArea.playfield_to_window
std::pair<double, double> playfield_to_screen(const WindowGeometry& g, double px, double py);
