cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(cuda_ocl_mem)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CUDA QUIET REQUIRED)
find_package(OpenCL REQUIRED)

enable_language("CUDA")
add_executable(cuda_ocl_mem main.cpp)
target_link_libraries(cuda_ocl_mem OpenCL::OpenCL ${CUDA_LIBRARIES})

install(TARGETS cuda_ocl_mem RUNTIME DESTINATION bin)
