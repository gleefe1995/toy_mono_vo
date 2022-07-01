cmake_minimum_required(VERSION 3.10)
project(toy_mono_vo)

# Should be larger than version 14 to avoid ‘integer_sequence’ is not a member of ‘std’ error
set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)

include_directories(
    include
    ${Opencv_INCLUDE_DIRS}
)
# Only Ceres itself should be compiled with CERES_BUILDING_SHARED_LIBRARY
# defined, any users of Ceres will have CERES_USING_SHARED_LIBRARY defined
# for them in Ceres' config.h if appropriate.

add_executable(classtest main.cpp)
target_link_libraries(classtest ${OpenCV_LIBS})

