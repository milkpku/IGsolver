cmake_minimum_required(VERSION 2.8)

project(Chol_test)

file(GLOB chol_list *.cpp)

add_executable(${PROJECT_NAME} ${chol_list})

target_link_libraries(${PROJECT_NAME} IGsolver)
