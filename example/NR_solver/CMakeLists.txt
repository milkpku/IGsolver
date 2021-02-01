cmake_minimum_required(VERSION 2.8)

project(NR_test)

file(GLOB NR_list *.cpp)

add_executable(${PROJECT_NAME} ${NR_list})

target_link_libraries(${PROJECT_NAME} IGsolver)
