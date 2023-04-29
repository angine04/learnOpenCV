cmake_minimum_required(VERSION 3.0.2)
project(sphereDetect)


add_executable(sphereDetect  sphereDetect.cpp)

find_package(OpenCV)

include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 11)

TARGET_LINK_LIBRARIES(sphereDetect ${OpenCV_LIBS})
