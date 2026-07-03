# Create the test runner executable target
add_executable(unit_tests 
    test_main.cpp
)

# Link the test executable to our core engine and GoogleTest components
target_link_libraries(unit_tests PRIVATE 
    vecdb_core
    gtest_main
    nlohmann_json::nlohmann_json
)

# Inject compile and link flags
target_compile_options(unit_tests PRIVATE ${COMPILER_FLAGS})
target_link_options(unit_tests PRIVATE ${LINKER_FLAGS})

target_compile_definitions(unit_tests PRIVATE TEST_DATA_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/test_data\")