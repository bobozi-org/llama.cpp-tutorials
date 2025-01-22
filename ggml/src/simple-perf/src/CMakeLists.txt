add_library(simple-perf simple_perf.cpp)
target_include_directories(simple-perf PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_options(simple-perf PRIVATE -fPIC)
