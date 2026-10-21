#pragma once

#include <windows.h>

#include <string>

// Loads capture_hook.dll (next to this executable) into the process that owns `window`. Returns an empty string on success, else why not.
std::string inject_hook(HWND window);

// Asks the hook to restore the swapchain and then unloads it from the process
std::string eject_hook(HWND window);
