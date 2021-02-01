cmake_minimum_required(VERSION 2.8)

project(GN_test)

file(GLOB GN_list *.cpp)

add_executable(${PROJECT_NAME} ${GN_list})

target_link_libraries(${PROJECT_NAME} IGsolver)
