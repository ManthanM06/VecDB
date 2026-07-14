# Define the core database engine as a static library target
add_library(vecdb_core 
    core.cpp
)

# Define where header files live. 
# INTERFACE means targets linking to this library will automatically know where these headers are.
target_include_directories(vecdb_core PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Phase 7: core.hpp now includes nlohmann/json.hpp — link it PUBLIC so that
# any target linking vecdb_core also inherits the json include path.
target_link_libraries(vecdb_core PUBLIC nlohmann_json::nlohmann_json)

# Inject the compiler options we defined in the root file
target_compile_options(vecdb_core PRIVATE ${COMPILER_FLAGS})
target_link_options(vecdb_core PRIVATE ${LINKER_FLAGS})