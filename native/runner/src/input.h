#pragma once

#include <windows.h>

#include <cmath>
#include <cstdint>

// SetCursorPos for the cursor and scancode keyboard events for the keys: synthetic mouse buttons are rejected on some machines
class Injector {
public:
    void move(double x, double y) const { SetCursorPos(static_cast<int>(std::nearbyint(x)), static_cast<int>(std::nearbyint(y))); }

    void key(uint16_t scan, bool down) const {
        INPUT in{};
        in.type = INPUT_KEYBOARD;
        in.ki.wScan = scan;
        in.ki.dwFlags = KEYEVENTF_SCANCODE | (down ? 0 : KEYEVENTF_KEYUP);
        SendInput(1, &in, sizeof(INPUT));
    }

    static uint16_t scan_code(char c) {
        const UINT vk = VkKeyScanW(static_cast<wchar_t>(c)) & 0xFF;
        return static_cast<uint16_t>(MapVirtualKeyW(vk, MAPVK_VK_TO_VSC));
    }
};
