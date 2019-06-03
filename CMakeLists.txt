cmake_minimum_required(VERSION 3.0)
project(dense_tracking)
add_executable(dense_tracking main.cpp)

SET(CMAKE_CXX_FLAGS "-std=c++11 -g -O3")

find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
	include_directories(${OpenCV_INCLUDE_DIRS})
	target_link_libraries(dense_tracking ${OpenCV_LIBS})
endif()
