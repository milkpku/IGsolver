cmake_minimum_required(VERSION 2.8)

project(IGsolver)
include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB solver_list ${PROJECT_SOURCE_DIR}/src/*.cpp)
add_library(IGsolver ${solver_list})

add_subdirectory(example/Chol_solver/)
add_subdirectory(example/SQP_solver/)
