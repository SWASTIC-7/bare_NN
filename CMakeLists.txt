cmake_minimum_required(VERSION 3.18)

project(bare_NN 
    VERSION 1.0.0
    DESCRIPTION "Bare metal neural network implementation using CUDA/PTX"
    LANGUAGES CXX CUDA
)

option(BUILD_TESTS      "Build test executables"           ON)
option(BUILD_EXAMPLES   "Build example executables"        ON)
option(GENERATE_PTX     "Generate PTX files from .cu"      OFF)
option(ENABLE_FAST_MATH "Enable fast math optimizations"   ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
endif()

message(STATUS "CUDA Toolkit Version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

set(CUDA_COMMON_FLAGS 
    --expt-relaxed-constexpr
    -Xcompiler=-Wall
)

if(ENABLE_FAST_MATH)
    list(APPEND CUDA_COMMON_FLAGS --use_fast_math)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    list(APPEND CUDA_COMMON_FLAGS -G -g -O0)
else()
    list(APPEND CUDA_COMMON_FLAGS -O3 -lineinfo)
endif()

add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMMON_FLAGS}>")

set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(SRC_DIR ${PROJECT_ROOT}/src)
set(TESTS_DIR ${PROJECT_ROOT}/tests)
set(PTX_DIR ${PROJECT_ROOT}/ptx)
set(INCLUDE_DIR ${PROJECT_ROOT}/include)

file(MAKE_DIRECTORY ${INCLUDE_DIR})

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(PTXCompilation)
include(HelperFunctions)

add_subdirectory(src)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(GENERATE_PTX)
    add_custom_target(generate_ptx ALL
        COMMENT "Generating PTX files"
    )
    
    generate_ptx_from_cuda(
        SOURCE ${TESTS_DIR}/vector_operation.cu
        OUTPUT ${PTX_DIR}/vector_operation_gen.ptx
        DEPENDS generate_ptx
    )
endif()

install(DIRECTORY ${PTX_DIR}/
    DESTINATION share/bare_nn/ptx
    FILES_MATCHING PATTERN "*.ptx"
)

install(DIRECTORY ${INCLUDE_DIR}/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.cuh"
)

message(STATUS "")
message(STATUS "=== bare_NN Build Configuration ===")
message(STATUS "  Build Type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA Toolkit:     ${CUDAToolkit_VERSION}")
message(STATUS "  CUDA Arch:        ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Build Tests:      ${BUILD_TESTS}")
message(STATUS "  Build Examples:   ${BUILD_EXAMPLES}")
message(STATUS "  Generate PTX:     ${GENERATE_PTX}")
message(STATUS "  Fast Math:        ${ENABLE_FAST_MATH}")
message(STATUS "===================================")
message(STATUS "")
