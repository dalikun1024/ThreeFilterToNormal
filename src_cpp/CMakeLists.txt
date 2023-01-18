cmake_minimum_required(VERSION 3.10)
project(depth_to_mesh)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS -O3)

set(src_list ${CMAKE_CURRENT_SOURCE_DIR}/depth_to_mesh.cpp
             ${CMAKE_CURRENT_SOURCE_DIR}/depth_to_mesh.hpp
             ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(depth_to_mesh ${src_list})

target_link_libraries(depth_to_mesh ${OpenCV_LIBS})

