cmake_minimum_required(VERSION 3.20)
project(hawkmoth_hover LANGUAGES C CXX)

find_package(IBAMR 0.19.0 CONFIG REQUIRED)

add_executable(hawkmoth_hover src/main.cpp)
target_link_libraries(hawkmoth_hover PRIVATE IBAMR::IBAMR3d)
target_compile_features(hawkmoth_hover PRIVATE cxx_std_17)

# IBAMR's installed library does not install test executables. Keep the exact
# upstream regression source and input/output alongside the benchmark.
set(IBAMR_REGRESSION_DIR "${CMAKE_CURRENT_SOURCE_DIR}/reference_data/ibamr_0.19.0_navier_stokes")
add_executable(ibamr_navier_stokes_01_3d "${IBAMR_REGRESSION_DIR}/navier_stokes_01.cpp")
target_link_libraries(ibamr_navier_stokes_01_3d PRIVATE IBAMR::IBAMR3d)
target_compile_features(ibamr_navier_stokes_01_3d PRIVATE cxx_std_17)
target_compile_definitions(ibamr_navier_stokes_01_3d PRIVATE NDIM=3)
