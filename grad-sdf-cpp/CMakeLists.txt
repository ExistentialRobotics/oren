cmake_minimum_required(VERSION 3.20)

project(grad_sdf_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c
          "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE TORCH_PATH_RESULT)

if(NOT TORCH_PATH_RESULT EQUAL 0)
  message(
    FATAL_ERROR
      "Could not get LibTorch path from Python. Is torch installed? Try: pip install torch"
  )
endif()

list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX})
find_package(Torch REQUIRED)
find_package(yaml-cpp REQUIRED)

add_library(grad_sdf_cpp src/config.cpp src/octree_module.cpp
                         src/residual_net_module.cpp src/sdf_network_module.cpp)

target_include_directories(grad_sdf_cpp
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(grad_sdf_cpp PUBLIC ${TORCH_LIBRARIES} yaml-cpp)

target_compile_features(grad_sdf_cpp PUBLIC cxx_std_17)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(grad_sdf_cpp PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(grad_sdf_infer src/main.cpp)
target_link_libraries(grad_sdf_infer PRIVATE grad_sdf_cpp)
target_compile_definitions(
  grad_sdf_infer PRIVATE GRAD_SDF_CPP_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(grad_sdf_eval src/eval.cpp)
target_link_libraries(grad_sdf_eval PRIVATE grad_sdf_cpp)

add_executable(extract_mesh src/extract_mesh.cpp)
target_link_libraries(extract_mesh PRIVATE grad_sdf_cpp)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set_property(TARGET grad_sdf_infer PROPERTY BUILD_RPATH "$ORIGIN")
endif()
