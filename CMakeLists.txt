cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(cuda_ocl_mem)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  # Force to always compile with W4
  if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
  endif()
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  # Update if necessary
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic")
endif()

find_package(CUDA QUIET REQUIRED)
find_package(OpenCL REQUIRED)

enable_language("CUDA")
add_executable(cuda_ocl_mem main.cpp)
target_link_libraries(cuda_ocl_mem OpenCL::OpenCL ${CUDA_LIBRARIES})

install(TARGETS cuda_ocl_mem RUNTIME DESTINATION bin)
