# Runner

The C++ code of the native runner: capture, ONNX inference, input and the Present hook. To use the runner, see `../README.md`.

```
src/               main.cpp (commands), player.cpp (100 Hz loop), processor.cpp (GPU gray and downsample),
                   wgc_source.cpp, hook.cpp (capture_hook.dll), hook_source.cpp, hook_inject.cpp, post.h, model.h
recipes/           local silo recipes: onnxruntime, cudnn, libcublas (prebuilt NVIDIA and Microsoft binaries)
silopkg.py         the silo package: dependencies and the CMake build
CMakeLists.txt     two targets, runner and capture_hook
```

## Build

```
cd play/runner
uv sync
uv run silo sync .
uv run silo run -C . -- cmake --preset runner-release
uv run silo run -C . -- cmake --build --preset runner-release
```

The output is `build/release/runner.exe` with `capture_hook.dll` and every runtime DLL beside it. `uv run silo build .` puts the same set in silo's store, which is what the `../shell` project runs.

## Notes

- Recipe files must use LF line endings (silo refuses CRLF), `.gitattributes` handles that.
- onnxruntime, cuDNN and cuBLAS are fetched by checksum from Microsoft and NVIDIA. The runtime DLLs are copied beside the exe by `CMakeLists.txt`, ONNX Runtime loads its providers by name at run time.
- The Python side that exports models and builds test fixtures is in `py/src/osu_auto` (`export_onnx.py`, `make_runner_fixture.py`, `make_shader_fixture.py`).
