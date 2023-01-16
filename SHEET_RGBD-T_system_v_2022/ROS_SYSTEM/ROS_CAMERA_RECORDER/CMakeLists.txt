cmake_minimum_required(VERSION 3.0.2)
project(thermal_camera_reader)

add_compile_options(-std=c++11)
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_MODULE_PATH "${thermal_camera_reader_SOURCE_DIR}/cmake")
include(build_type)

set (WITH_ADDRESS_SANITIZER false CACHE BOOL "Enable address sanitizer")
set (WITH_DEBUG_VERBOSITY false CACHE BOOL "Enable verbose stdout messages")
set (default_build_type "Release")

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

#if (WITH_DEBUG_VERBOSITY)
#    add_definitions(-DSEEK_DEBUG)
#endif ()

#message (STATUS "Build type: ${CMAKE_BUILD_TYPE}")

#include (MacroLogFeature)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages

find_package(catkin REQUIRED
  COMPONENTS
  roscpp
  #rospy
  #dynamic_reconfigure
  geometry_msgs
  #message_generation
  std_msgs
  sensor_msgs
  cv_bridge
  #tf2_ros
)


find_package (LibUSB)
#macro_log_feature (LIBUSB_FOUND "libusb" "Required to communicate via USB" "https://libusb.info/" TRUE)

find_package (OpenCV REQUIRED)
#macro_log_feature (OPENCV_FOUND "OpenCV" "Required to handle image processing" "https://opencv.org/" TRUE)




## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a exec_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a exec_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   sensor_msgs
# )

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a exec_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )



catkin_package(
  INCLUDE_DIRS include
  LIBRARIES thermal_camera_reader
  CATKIN_DEPENDS  roscpp
                  std_msgs
                  geometry_msgs
                  sensor_msgs
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${LIBUSB_INCLUDE_DIRS}
  ${OPENCV_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
  src/SeekCam.cpp
  src/SeekDevice.cpp
  src/SeekThermal.cpp
  src/SeekThermalPro.cpp
  src/thermal_camera_reader.cpp
)
target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${LIBUSB_LIBRARIES}
  ${usb_LIBRARY}
  ${LIBUSB_DIR}
  ${OPENCV_LIBRARIES}
)

add_executable(${PROJECT_NAME}_node src/thermal_camera_reader_node.cpp)
target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  ${LIBUSB_LIBRARIES}
  ${usb_LIBRARY}
  ${LIBUSB_DIR}
  ${OPENCV_LIBRARIES}
  ${PROJECT_NAME}
)

