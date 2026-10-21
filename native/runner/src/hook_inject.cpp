#include "hook_inject.h"

#include <tlhelp32.h>

#include <filesystem>

#include "hook_shm.h"

namespace {

struct Handle {
    HANDLE h;
    explicit Handle(HANDLE handle) : h(handle) {}
    ~Handle() {
        if (h) CloseHandle(h);
    }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
};

std::filesystem::path dll_path() {
    wchar_t exe[MAX_PATH];
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    return std::filesystem::path(exe).parent_path() / kHookDllName;
}

HANDLE open_target(HWND window) {
    DWORD pid = 0;
    GetWindowThreadProcessId(window, &pid);
    return OpenProcess(PROCESS_CREATE_THREAD | PROCESS_QUERY_INFORMATION | PROCESS_VM_OPERATION | PROCESS_VM_WRITE | PROCESS_VM_READ, FALSE, pid);
}

// Runs `function(argument)` in the target and returns the low 32 bits of its result
std::string run_remote(HANDLE process, LPTHREAD_START_ROUTINE function, LPVOID argument, DWORD* result) {
    Handle thread(CreateRemoteThread(process, nullptr, 0, function, argument, 0, nullptr));
    if (!thread.h) return "CreateRemoteThread failed: " + std::to_string(GetLastError());
    if (WaitForSingleObject(thread.h, 15000) != WAIT_OBJECT_0) return "remote thread timed out";
    DWORD code = 0;
    GetExitCodeThread(thread.h, &code);
    if (result) *result = code;
    return "";
}

std::string remote_module_base(HWND window, BYTE** base) {
    DWORD pid = 0;
    GetWindowThreadProcessId(window, &pid);
    Handle snapshot(CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, pid));
    if (snapshot.h == INVALID_HANDLE_VALUE) return "cannot list modules";
    MODULEENTRY32W entry{sizeof(entry)};
    for (BOOL ok = Module32FirstW(snapshot.h, &entry); ok; ok = Module32NextW(snapshot.h, &entry)) {
        if (_wcsicmp(entry.szModule, kHookDllName) == 0) {
            *base = entry.modBaseAddr;
            return "";
        }
    }
    return "hook is not loaded in that process";
}

}  // namespace

std::string inject_hook(HWND window) {
    const std::wstring path = dll_path().wstring();
    if (!std::filesystem::exists(path)) return "missing " + dll_path().string();
    BYTE* existing = nullptr;
    if (remote_module_base(window, &existing).empty()) return "hook is already loaded";

    Handle process(open_target(window));
    if (!process.h) return "OpenProcess failed: " + std::to_string(GetLastError());
    const SIZE_T bytes = (path.size() + 1) * sizeof(wchar_t);
    void* memory = VirtualAllocEx(process.h, nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!memory) return "VirtualAllocEx failed";
    WriteProcessMemory(process.h, memory, path.c_str(), bytes, nullptr);
    DWORD loaded = 0;
    const std::string error = run_remote(process.h, reinterpret_cast<LPTHREAD_START_ROUTINE>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "LoadLibraryW")), memory, &loaded);
    VirtualFreeEx(process.h, memory, 0, MEM_RELEASE);
    if (!error.empty()) return error;
    return loaded ? "" : "LoadLibraryW returned null in the target";
}

std::string eject_hook(HWND window) {
    BYTE* remote = nullptr;
    if (auto error = remote_module_base(window, &remote); !error.empty()) return error;
    Handle process(open_target(window));
    if (!process.h) return "OpenProcess failed: " + std::to_string(GetLastError());

    HMODULE local = LoadLibraryExW(dll_path().c_str(), nullptr, DONT_RESOLVE_DLL_REFERENCES);
    if (!local) return "cannot load the hook locally to find its exports";
    const auto offset = reinterpret_cast<BYTE*>(GetProcAddress(local, "HookEject")) - reinterpret_cast<BYTE*>(local);
    FreeLibrary(local);

    if (auto error = run_remote(process.h, reinterpret_cast<LPTHREAD_START_ROUTINE>(remote + offset), nullptr, nullptr); !error.empty()) return error;
    return run_remote(process.h, reinterpret_cast<LPTHREAD_START_ROUTINE>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "FreeLibrary")), remote, nullptr);
}
