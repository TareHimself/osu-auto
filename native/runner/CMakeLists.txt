cmake_minimum_required(VERSION 4.0)
project(osu_auto_runner CXX)

find_package(onnxruntime REQUIRED)
find_package(cudnn REQUIRED)
find_package(libcublas REQUIRED)

add_executable(runner src/main.cpp src/processor.cpp src/window.cpp src/wgc_source.cpp src/player.cpp src/hook_source.cpp src/hook_inject.cpp)
target_compile_features(runner PRIVATE cxx_std_20)
target_compile_definitions(runner PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_link_libraries(runner PRIVATE onnxruntime::onnxruntime d3d11 dxgi d3dcompiler dwmapi windowsapp)
install(TARGETS runner RUNTIME DESTINATION bin)

add_library(capture_hook SHARED src/hook.cpp src/processor.cpp src/window.cpp)
target_compile_features(capture_hook PRIVATE cxx_std_20)
target_compile_definitions(capture_hook PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
set_target_properties(capture_hook PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded")
target_link_libraries(capture_hook PRIVATE d3d11 dxgi d3dcompiler dwmapi user32)
add_dependencies(runner capture_hook)
install(TARGETS capture_hook RUNTIME DESTINATION bin)

# ONNX Runtime loads its providers and their CUDA libraries by name at run time, so every DLL goes beside the executable
file(GLOB runtime_dlls ${onnxruntime_ROOT_DIR}/bin/*.dll ${cudnn_ROOT_DIR}/bin/*.dll ${libcublas_ROOT_DIR}/bin/*.dll)
add_custom_command(TARGET runner POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${runtime_dlls} $<TARGET_FILE_DIR:runner>)
install(FILES ${runtime_dlls} DESTINATION bin)
