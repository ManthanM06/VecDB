add_executable(core_bench bench_core.cpp)

target_link_libraries(core_bench PRIVATE 
    vecdb_core
    benchmark::benchmark
)

target_compile_options(core_bench PRIVATE ${COMPILER_FLAGS})
target_link_options(core_bench PRIVATE ${LINKER_FLAGS})