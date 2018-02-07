cmake_minimum_required(VERSION 2.8)

project(IGsolver)
include_directories(include/IGsolver)

file(GLOB solver_list include/IGsolver/*.cpp)
add_library(IGsolver SHARED ${solver_list})

add_subdirectory(example/chol_solver/)
