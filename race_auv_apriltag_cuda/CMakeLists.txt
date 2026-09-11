cmake_minimum_required(VERSION 3.16)
project(race_auv_apriltag_cuda)

find_package(ament_cmake REQUIRED)

# CUDA is optional. On non-Jetson hosts (e.g. the sim computer) the
# package still configures and installs, just without the native shims;
# the node then uses its CPU backends (detector_backend: "python",
# image_pipeline: "cpu").
find_package(CUDAToolkit QUIET)

# Root of the vendored cuAprilTags tree. The package sits next to
# third_party/ at the race_auv_perception repository root, so the default
# is a sibling of this package. Override with -DCUAPRILTAGS_ROOT=... when
# the submodule lives elsewhere.
set(CUAPRILTAGS_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/isaac_ros_nitros/isaac_ros_nitros/lib/cuapriltags"
    CACHE PATH "Root of the cuapriltags directory (contains cuapriltags/ and lib_*/)")

set(_install_targets "")

if(NOT CUDAToolkit_FOUND)
    message(WARNING
        "race_auv_apriltag_cuda: CUDAToolkit not found; building without the "
        "CUDA apriltag/image shims. The CPU Python backend remains available.")
else()
    # Pick the prebuilt library that matches the host architecture. NVIDIA
    # only publishes these two variants in isaac_ros_nitros release-4.6.
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        set(_cuapriltags_libdir "lib_aarch64_jetpack61")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
        set(_cuapriltags_libdir "lib_x86_64_cuda_12_6")
    else()
        set(_cuapriltags_libdir "")
        message(WARNING
            "race_auv_apriltag_cuda: unsupported architecture "
            "'${CMAKE_SYSTEM_PROCESSOR}'; skipping the CUDA shims.")
    endif()

    if(_cuapriltags_libdir)
        set(CUAPRILTAGS_LIBRARY
            "${CUAPRILTAGS_ROOT}/${_cuapriltags_libdir}/libcuapriltags.a")
        if(NOT EXISTS "${CUAPRILTAGS_LIBRARY}")
            message(WARNING
                "race_auv_apriltag_cuda: cuAprilTags library not found at:\n"
                "  ${CUAPRILTAGS_LIBRARY}\n"
                "Run 'scripts/setup_third_party.sh' from the "
                "race_auv_perception repository root to fetch the "
                "isaac_ros_nitros submodule and its Git LFS objects. "
                "Skipping the CUDA detector shim.")
        else()
            add_library(race_auv_apriltag_cuda SHARED src/cuapriltags_shim.cpp)
            target_compile_features(race_auv_apriltag_cuda PRIVATE cxx_std_17)
            target_include_directories(race_auv_apriltag_cuda PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}/include"
                "${CUAPRILTAGS_ROOT}/cuapriltags"
                ${CUDAToolkit_INCLUDE_DIRS}
            )
            target_link_libraries(race_auv_apriltag_cuda
                "${CUAPRILTAGS_LIBRARY}"
                CUDA::cudart
            )
            set_target_properties(race_auv_apriltag_cuda PROPERTIES
                POSITION_INDEPENDENT_CODE ON
                LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib"
            )
            list(APPEND _install_targets race_auv_apriltag_cuda)
        endif()
    endif()

    # -----------------------------------------------------------------------
    # GPU image stages (nvjpeg decode/encode + VPI CUDA fisheye rectify).
    # VPI ships with JetPack; its CMake package lives outside the default
    # prefix. On hosts without VPI (e.g. an x86 dev box) the library is
    # skipped and the node falls back to the CPU image pipeline.
    # -----------------------------------------------------------------------
    list(APPEND CMAKE_PREFIX_PATH "/opt/nvidia/vpi4/lib/${CMAKE_LIBRARY_ARCHITECTURE}")
    find_package(vpi QUIET)

    if(vpi_FOUND AND TARGET CUDA::nvjpeg AND _install_targets)
        add_library(race_auv_image_cuda SHARED src/image_cuda_shim.cpp)
        target_compile_features(race_auv_image_cuda PRIVATE cxx_std_17)
        target_include_directories(race_auv_image_cuda PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            ${CUDAToolkit_INCLUDE_DIRS}
        )
        target_link_libraries(race_auv_image_cuda
            vpi
            CUDA::nvjpeg
            CUDA::cudart
        )
        set_target_properties(race_auv_image_cuda PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib"
        )
        list(APPEND _install_targets race_auv_image_cuda)
        message(STATUS "race_auv_apriltag_cuda: VPI found, building race_auv_image_cuda")
    else()
        message(WARNING
            "race_auv_apriltag_cuda: VPI/nvjpeg or the CUDA detector shim "
            "unavailable; race_auv_image_cuda (nvjpeg/VPI GPU image stages) "
            "will not be built.")
    endif()
endif()

# Install the shims where Python resolves them through
# ament_index_python.get_package_share_directory(). No LD_LIBRARY_PATH needed.
if(_install_targets)
    install(TARGETS ${_install_targets}
        LIBRARY DESTINATION share/${PROJECT_NAME}/lib)
endif()
install(DIRECTORY include/
    DESTINATION include)

ament_export_include_directories(include)

ament_package()
