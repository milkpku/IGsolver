cmake_minimum_required(VERSION 2.8)

project(IGsolver)
include_directories(include/IGsolver)

file(GLOB solver_list include/IGsolver/*.cpp)
add_library(IGsolver ${solver_list})

add_subdirectory(example/Chol_solver/)
add_subdirectory(example/SQP_solver/)
