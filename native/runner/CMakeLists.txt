cmake_minimum_required(VERSION 4.0)
project(osu_auto_runner CXX)

find_package(onnxruntime REQUIRED)

add_executable(runner src/main.cpp src/processor.cpp src/window.cpp src/wgc_source.cpp src/player.cpp)
target_compile_features(runner PRIVATE cxx_std_20)
target_compile_definitions(runner PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_link_libraries(runner PRIVATE onnxruntime::onnxruntime d3d11 dxgi d3dcompiler dwmapi windowsapp)
install(TARGETS runner RUNTIME DESTINATION bin)

set(runtime_dlls $<TARGET_RUNTIME_DLLS:runner>)
add_custom_command(TARGET runner POST_BUILD
  COMMAND "$<$<BOOL:${runtime_dlls}>:${CMAKE_COMMAND};-E;copy_if_different;${runtime_dlls};$<TARGET_FILE_DIR:runner>>"
  COMMAND_EXPAND_LISTS)

get_target_property(ort_dll onnxruntime::onnxruntime IMPORTED_LOCATION_RELEASE)
if(NOT ort_dll)
  get_target_property(ort_dll onnxruntime::onnxruntime IMPORTED_LOCATION)
endif()
get_filename_component(ort_bin ${ort_dll} DIRECTORY)
file(GLOB ort_provider_dlls ${ort_bin}/onnxruntime_providers_*.dll)
add_custom_command(TARGET runner POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ort_provider_dlls} $<TARGET_FILE_DIR:runner>)
