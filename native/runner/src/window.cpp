#include "window.h"

#include <dwmapi.h>

#include <algorithm>
#include <cctype>
#include <cwctype>
#include <cmath>
#include <vector>

namespace {

struct Search {
    std::wstring prefix;
    std::optional<HWND> found;
};

std::wstring lower(std::wstring s) {
    std::transform(s.begin(), s.end(), s.begin(), [](wchar_t c) { return static_cast<wchar_t>(std::towlower(c)); });
    return s;
}

BOOL CALLBACK visit(HWND hwnd, LPARAM param) {
    auto* search = reinterpret_cast<Search*>(param);
    if (!IsWindowVisible(hwnd)) return TRUE;
    wchar_t title[256];
    const int n = GetWindowTextW(hwnd, title, 256);
    if (n > 0 && lower(std::wstring(title, n)).starts_with(search->prefix)) {
        search->found = hwnd;
        return FALSE;
    }
    return TRUE;
}

}  // namespace

std::optional<HWND> find_window(const std::wstring& title_prefix) {
    Search search{lower(title_prefix), std::nullopt};
    EnumWindows(visit, reinterpret_cast<LPARAM>(&search));
    return search.found;
}

// Same as PlayArea.capture rounded: 4:3 box centred in the client area, Python rounds halves to even like nearbyint
Crop capture_box(int client_w, int client_h, int offset_x, int offset_y) {
    const double box_w = std::min<double>(client_w, client_h * 4.0 / 3.0);
    const double box_h = box_w * 3.0 / 4.0;
    return {offset_x + static_cast<int>(std::nearbyint((client_w - box_w) / 2)), offset_y + static_cast<int>(std::nearbyint((client_h - box_h) / 2)),
            static_cast<int>(std::nearbyint(box_w)), static_cast<int>(std::nearbyint(box_h))};
}

WindowGeometry window_geometry(HWND hwnd) {
    RECT client;
    GetClientRect(hwnd, &client);
    POINT origin{client.left, client.top};
    ClientToScreen(hwnd, &origin);
    RECT bounds{};
    DwmGetWindowAttribute(hwnd, DWMWA_EXTENDED_FRAME_BOUNDS, &bounds, sizeof(bounds));

    const int w = client.right - client.left, h = client.bottom - client.top;
    return {w, h, origin.x, origin.y, capture_box(w, h, origin.x - bounds.left, origin.y - bounds.top)};
}
