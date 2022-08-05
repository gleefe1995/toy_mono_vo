# cmake_minimum_required(VERSION 3.10)
cmake_minimum_required(VERSION 3.0)
SET(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(toy_mono_vo LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE "Release")

# Should be larger than version 14 to avoid ‘integer_sequence’ is not a member of ‘std’ error
# set(CMAKE_CXX_STANDARD 17)

### OpenMP
find_package(OpenMP)
include_directories(${OpenMP_INCLUDE_DIRS})
###

### Torchlib
find_package(Torch REQUIRED PATHS ${PROJECT_SOURCE_DIR}/libtorch)
# set(CMAKE_PREFIX_PATH "/home/gleefe/ALIKE-cpp/libtorch/share/cmake/Torch")
# find_package(Torch REQUIRED)
message(STATUS "Found Torch include is ${TORCH_INCLUDE_DIRS}")
include_directories(${TORCH_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Wno-deprecated-gpu-targets")
###


find_package(OpenCV REQUIRED)
# find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Ceres REQUIRED)

include_directories(
    include
    ${Opencv_INCLUDE_DIRS}
    # ${EIGEN3_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
)
# Only Ceres itself should be compiled with CERES_BUILDING_SHARED_LIBRARY
# defined, any users of Ceres will have CERES_USING_SHARED_LIBRARY defined
# for them in Ceres' config.h if appropriate.

cuda_add_library(get_patches_cuda SHARED src/get_patches.cu)
set_target_properties(get_patches_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_library(get_patches SHARED src/get_patches.cpp)
target_link_libraries(get_patches get_patches_cuda)

add_executable(classtest 
                src/main.cpp
                src/utils.cpp)
target_link_libraries(classtest ${TORCH_LIBRARIES} ${OpenCV_LIBS} ${Pangolin_LIBRARIES} ${CERES_LIBRARIES} get_patches)

# ${EIGEN3_LIBS}

