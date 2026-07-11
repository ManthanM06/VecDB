cmake_minimum_required(VERSION 3.24)
project(VecDB VERSION 1.0 LANGUAGES CXX)

# Force Modern C++20 Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Generate compile_commands.json for clangd autocomplete
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Enforce strict compiler warnings to catch errors early
set(GLOBAL_COMPILER_WARNINGS 
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wunused
)

# Apply global build types (Debug vs Release)
if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Configuring Debug Build with Sanitizers...")
    set(COMPILER_FLAGS ${GLOBAL_COMPILER_WARNINGS} -g -O0 -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer -march=native) 
    set(LINKER_FLAGS -fsanitize=address -fsanitize=undefined)
else()
    message(STATUS "Configuring Release Build with Optimizations...")
    set(COMPILER_FLAGS ${GLOBAL_COMPILER_WARNINGS} -O3 -march=native -DNDEBUG -flto)
    set(LINKER_FLAGS -flto)
endif()

# Fetch GoogleTest safely via modern FetchContent
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Fetch nlohmann/json
# c++ does not have built-in JSON parser hence this 
FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(nlohmann_json)

#fetch google benchmark
FetchContent_Declare(
  googlebenchmark
  URL https://github.com/google/benchmark/archive/refs/tags/v1.8.3.zip
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

# cpp-httplib (REST Server)
FetchContent_Declare(
    httplib
    URL https://github.com/yhirose/cpp-httplib/archive/refs/tags/v0.15.3.zip
)
FetchContent_MakeAvailable(httplib)

# Descend into modular subdirectories
add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(benchmarks)

# Add the server executable
add_executable(vecdb_server src/server.cpp)

# Link the core engine and the new networking libraries
target_link_libraries(vecdb_server PRIVATE 
    vecdb_core
    httplib
    nlohmann_json::nlohmann_json
)

target_compile_options(vecdb_server PRIVATE ${COMPILER_FLAGS})
target_link_options(vecdb_server PRIVATE ${LINKER_FLAGS})