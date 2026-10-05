cmake_minimum_required(VERSION 3.16)
find_package(ASC REQUIRED)
project(bmmms_npu_validation LANGUAGES ASC CXX)
set(CMAKE_CXX_STANDARD 14)
if(NOT DEFINED NPU_ARCH)
    set(NPU_ARCH "dav-2201")
endif()
add_executable(bmmms_npu_runner npu_runner.asc)
target_compile_options(bmmms_npu_runner PRIVATE $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH}>)
option(BMMMS_TRACE_RUNTIME "Trace synchronization inside the submitted Host entry" OFF)
if(BMMMS_TRACE_RUNTIME)
    target_compile_definitions(bmmms_npu_runner PRIVATE BMMMS_TRACE_RUNTIME)
endif()
target_link_libraries(bmmms_npu_runner PRIVATE tiling_api register platform unified_dlog dl m graph_base)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/../kernel.asc" "${CMAKE_CURRENT_SOURCE_DIR}/npu_runner.asc")
# Written only after a successful link, and bound to the resulting binary.
function(record_validation_build target)
set_property(TARGET ${target} APPEND PROPERTY LINK_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt" "${CMAKE_CURRENT_SOURCE_DIR}/write_build_manifest.cmake")
add_custom_command(TARGET ${target} POST_BUILD
    COMMAND ${CMAKE_COMMAND}
        "-DKERNEL=${CMAKE_CURRENT_SOURCE_DIR}/../kernel.asc"
        "-DRUNNER=${CMAKE_CURRENT_SOURCE_DIR}/npu_runner.asc"
        "-DBUILD_SPEC=${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
        "-DBINARY=$<TARGET_FILE:${target}>"
        "-DARCH=${NPU_ARCH}"
        "-DCOMPILER=${CMAKE_ASC_COMPILER}"
        "-DFLAGS=${CMAKE_ASC_FLAGS}|${CMAKE_BUILD_TYPE}|${CMAKE_ASC_FLAGS_RELEASE}|${CMAKE_ASC_FLAGS_DEBUG}|${CMAKE_ASC_FLAGS_RELWITHDEBINFO}|${CMAKE_ASC_FLAGS_MINSIZEREL}|trace=${BMMMS_TRACE_RUNTIME}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/write_build_manifest.cmake"
    VERBATIM)
endfunction()
record_validation_build(bmmms_npu_runner)
option(BMMMS_BUILD_SHARED_TEST "Validate standalone kernel shared-library loading" OFF)
if(BMMMS_BUILD_SHARED_TEST)
    add_library(bmmms_kernel SHARED kernel_library.asc)
    target_compile_options(bmmms_kernel PRIVATE $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH}>)
    target_link_options(bmmms_kernel PRIVATE -Wl,-z,defs)
    target_link_libraries(bmmms_kernel PRIVATE tiling_api register platform unified_dlog dl m graph_base)
    find_path(BMMMS_ACL_INCLUDE acl/acl.h PATHS "$ENV{ASCEND_HOME_PATH}/include" REQUIRED)
    find_library(BMMMS_ACL_LIBRARY ascendcl PATHS "$ENV{ASCEND_HOME_PATH}/lib64" REQUIRED)
    add_executable(bmmms_shared_runner shared_runner.cpp)
    target_include_directories(bmmms_shared_runner PRIVATE ${BMMMS_ACL_INCLUDE})
    target_link_libraries(bmmms_shared_runner PRIVATE ${BMMMS_ACL_LIBRARY} dl)
    record_validation_build(bmmms_kernel)
    record_validation_build(bmmms_shared_runner)
endif()
