# Define the core database engine as a static library target
add_library(vecdb_core 
    dummy.cpp
)

# Define where header files live. 
# INTERFACE means targets linking to this library will automatically know where these headers are.
target_include_directories(vecdb_core PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Inject the compiler options we defined in the root file
target_compile_options(vecdb_core PRIVATE ${COMPILER_FLAGS})
target_link_options(vecdb_core PRIVATE ${LINKER_FLAGS})