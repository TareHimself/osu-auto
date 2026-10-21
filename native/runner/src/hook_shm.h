#pragma once

#include <cstdint>

constexpr wchar_t kHookShmName[] = L"Local\\osu-auto-capture";
constexpr wchar_t kHookEventName[] = L"Local\\osu-auto-capture-frame";
constexpr wchar_t kHookDllName[] = L"capture_hook.dll";
constexpr uint32_t kHookMagic = 0x5043414F;  // "OACP"
constexpr uint32_t kHookVersion = 1;

// seq is odd while the hook is writing, so a reader keeps a copy only if seq was even and unchanged around it
struct HookShm {
    uint32_t magic;
    uint32_t version;
    uint32_t width;
    uint32_t height;
    volatile uint64_t seq;
    volatile int64_t qpc;
    uint8_t frame[160 * 120];
};
