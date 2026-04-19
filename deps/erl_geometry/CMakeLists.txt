cmake_minimum_required(VERSION 3.22)

project(erl_geometry
    LANGUAGES CXX
    VERSION 0.1.0
)

# C++
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif ()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_VISIBILITY_PRESET "default")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fopenmp -Wall -Wextra -Wno-unknown-pragmas")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--disable-new-dtags") # pass arguments to the linker
# disable new DTAGS (DT_RUNPATH) since it is not supported in Ubuntu old DTAGS (DT_RPATH) is used to specify paths
# for libraries that are directly linked to the executable new DTAGS (DT_RUNPATH) is used to specify paths for
# libraries that are transitively linked to the executable
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color -fdiagnostics-show-template-tree")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftrack-macro-expansion=2")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -funroll-loops -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -funroll-loops -flto -ffat-lto-objects")
# -flto enables link-time optimization -ffat-lto-objects makes object files suitable for both LTO and non-LTO builds
find_program(CCACHE_FOUND ccache)

if (CCACHE_FOUND)
    set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
else ()
    message(STATUS "ccache is not found")
endif ()

if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DNDEBUG)
else ()
    set(CMAKE_VERBOSE_MAKEFILE ON)
endif ()

# CUDA
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE STRING "Host compiler for CUDA" FORCE)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
enable_language(CUDA)
set(CMAKE_CUDA_VISIBILITY_PRESET "default")
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_DEBUG "-G")

# dependencies
find_package(Eigen3 REQUIRED)
find_package(absl REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)
get_filename_component(TORCH_LIBRARIES_DIR "${Torch_DIR}/../../../lib" ABSOLUTE)

include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${Python3_INCLUDE_DIRS})

file(GLOB_RECURSE SRC_FILES src/*.cpp src/*.cu)
set(lib_name pyerl_geometry)

pybind11_add_module(${lib_name}
    ${SRC_FILES}
)

set_target_properties(${lib_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(${lib_name}
    PRIVATE Eigen3::Eigen absl::flat_hash_map absl::flat_hash_set yaml-cpp
    PRIVATE ${TORCH_LIBRARIES_DIR}/libtorch_python.so)
# pytorch comes with fmt
target_compile_definitions(${lib_name} PRIVATE ERL_USE_LIBTORCH ERL_USE_FMT PYBIND_MODULE_NAME=${lib_name})

if (APPLE)
    set(CMAKE_MACOSX_RPATH ON)
    set(_rpath_portable_origin "@loader_path")
else ()
    set(_rpath_portable_origin $ORIGIN)
endif ()
set(_rpath "${_rpath_portable_origin}:${_rpath_portable_origin}/lib/${PROJECT_NAME}")
set_target_properties(
    ${lib_name}
    PROPERTIES SKIP_BUILD_RPATH FALSE # Use separate rpaths during build and install phases
                BUILD_WITH_INSTALL_RPATH FALSE # Don't use the install-rpath during the build phase
                INSTALL_RPATH "${_rpath}" # search in the same directory, or in lib/<project_name>
                INSTALL_RPATH_USE_LINK_PATH TRUE)


if (DEFINED PIP_LIB_DIR)
    install(TARGETS ${lib_name} #
            ARCHIVE DESTINATION ${PIP_LIB_DIR} #
            LIBRARY DESTINATION ${PIP_LIB_DIR} #
            RUNTIME DESTINATION ${PIP_LIB_DIR})
endif()
