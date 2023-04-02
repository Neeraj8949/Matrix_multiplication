cmake_minimum_required(VERSION 3.16)

project(matrix_multiplication_hpx)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(HPX REQUIRED)

add_executable(matrix_multiplication matrix_multiplication.cpp)
target_link_libraries(matrix_multiplication PRIVATE HPX::hpx HPX::wrap_main)
