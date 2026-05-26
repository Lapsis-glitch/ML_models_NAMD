cmake_minimum_required(VERSION 3.18)
project(fennix_pjrt_shim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(fennix_pjrt_probe
    src/main.cpp
    src/pjrt_plugin.cpp
)

target_include_directories(fennix_pjrt_probe PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(fennix_pjrt_probe PRIVATE dl)

target_compile_options(fennix_pjrt_probe PRIVATE -Wall -Wextra -Wpedantic -fpermissive)
