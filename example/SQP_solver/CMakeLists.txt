cmake_minimum_required(VERSION 2.8)

project(SQP_test)

file(GLOB sqp_list *.cpp)

add_executable(${PROJECT_NAME} ${sqp_list})

target_link_libraries(${PROJECT_NAME} IGsolver)
