cmake_minimum_required(VERSION 3.10)
project(yolov10_ncnn_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find ncnn (assumes ncnn is installed system-wide or via CMAKE_PREFIX_PATH)
find_package(ncnn REQUIRED)

add_executable(yolov10_bench yolov10_bench.cpp)
target_link_libraries(yolov10_bench ncnn)
