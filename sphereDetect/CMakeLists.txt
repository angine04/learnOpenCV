cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 14)
project(sphereDetect)


add_executable(sphereDetect  sphereDetect.cpp)

find_package(OpenCV)
find_package(yaml-cpp)

find_package(PkgConfig)
pkg_check_modules(YAMLCPP REQUIRED yaml-cpp>=0.5)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${YAMLCPP_INCLUDE_DIRS})



TARGET_LINK_LIBRARIES(sphereDetect ${OpenCV_LIBS} ${YAMLCPP_LIBRARIES})
