cmake_minimum_required(VERSION 3.6)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()

project(CamViewer
    DESCRIPTION "Example QT application with fatigue detection"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "-Wall -Winvalid-pch -Wnon-virtual-dtor -Wextra -Wno-unused-parameter")

# Find dependencies
find_package(PkgConfig REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Qt5 COMPONENTS Widgets REQUIRED)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Find libcamera via pkg-config
pkg_check_modules(LIBCAMERA REQUIRED IMPORTED_TARGET libcamera)

# Optional: Qwt (already used by your QwtThermo)
pkg_check_modules(QWT REQUIRED IMPORTED_TARGET qwt-qt5)

# Print debug info
message(STATUS "libcamera library found:")
message(STATUS "    version: ${LIBCAMERA_VERSION}")
message(STATUS "    libraries: ${LIBCAMERA_LINK_LIBRARIES}")
message(STATUS "    include path: ${LIBCAMERA_INCLUDE_DIRS}")

message(STATUS "opencv:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")


add_subdirectory(libcam2opencv)


include_directories(${CMAKE_SOURCE_DIR}
                    ${LIBCAMERA_INCLUDE_DIRS}
                    ${OPENCV_INCLUDE_DIRS})

include(GNUInstallDirs)


add_executable(qtviewer
  main.cpp
  window.cpp
  fatigue_detector.cpp    
)

# Link libraries
target_link_libraries(qtviewer
  PkgConfig::LIBCAMERA
  Qt5::Widgets
  PkgConfig::QWT         
  ${OpenCV_LIBS}
  cam2opencv             
)
