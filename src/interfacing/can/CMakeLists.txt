cmake_minimum_required(VERSION 3.10)
project(can)

# Set compiler to C++ 17 Standard
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_index_cpp REQUIRED) # ROS tool to find shared directories of installed packages
find_package(ament_cmake REQUIRED) # ROS2 build tool
find_package(rclcpp REQUIRED) # ROS2 C++ package
find_package(std_msgs REQUIRED) # ROS2 message package
find_package(Boost REQUIRED) # Boost libraries

find_package(common_msgs REQUIRED)
find_package(yaml-cpp REQUIRED) # MIT protocol per-motor scaling constants (mit_profiles.yaml)

# Compiles source files into a library
# A library is not executed, instead other executables can link
# against it to access defined methods and classes.
# We build a library so that the methods defined can be used by
# both the unit test and ROS2 node executables.

add_library(can_lib
  src/can_core.cpp
  src/mit_protocol.cpp
)

# Indicate to compiler where to search for header files
target_include_directories(can_lib PRIVATE
  /usr/local/include
  PUBLIC
  include
)

# Add ROS2 dependencies required by package 
ament_target_dependencies(can_lib
  ament_index_cpp
  rclcpp
  std_msgs
  common_msgs
)

# Link system libraries for CAN support
# Note: Using system CAN libraries instead of libsocketcan for broader compatibility

target_link_libraries(can_lib
  /usr/local/lib/libdbcppp.so   
  ament_index_cpp::ament_index_cpp
)

# Create the can_node executable
add_executable(can_node src/can_node.cpp src/can_core.cpp src/mit_protocol.cpp)
ament_target_dependencies(can_node rclcpp std_msgs)

# Link to the previously built libraries
target_link_libraries(can_node can_lib yaml-cpp)

# Copy executable to installation location
install(TARGETS
  can_node
  DESTINATION lib/${PROJECT_NAME})

# Install launch and config files
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME})

# Install scripts with execute permissions
install(PROGRAMS
  scripts/setup_can.sh
  scripts/setup_socketcan.sh
  scripts/can_udev.sh
  scripts/calibrate_actuators.py
  scripts/telemetry_record.py
  scripts/roundtrip.py
  DESTINATION share/${PROJECT_NAME}/scripts
)
# Also expose the operator-facing scripts on PATH via lib/<pkg>
install(PROGRAMS
  scripts/calibrate_actuators.py
  scripts/telemetry_record.py
  scripts/roundtrip.py
  scripts/leg_angles.py
  DESTINATION lib/${PROJECT_NAME}
)
# Modules the scripts above import from their own directory
install(FILES
  scripts/joint_config.py
  scripts/telemetry.py
  DESTINATION share/${PROJECT_NAME}/scripts
)
install(FILES
  scripts/joint_config.py
  scripts/telemetry.py
  DESTINATION lib/${PROJECT_NAME}
)

# Testing and Development
find_package(ament_lint_auto REQUIRED)
ament_lint_auto_find_test_dependencies()

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  # MIT packing + the real humanoid.dbc layout (no ROS, no bus).
  ament_add_gtest(test_mit_protocol test/test_mit_protocol.cpp src/mit_protocol.cpp)
  target_include_directories(test_mit_protocol PRIVATE include /usr/local/include)
  target_link_libraries(test_mit_protocol /usr/local/lib/libdbcppp.so)
  target_compile_definitions(test_mit_protocol
    PRIVATE DBC_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../dbc/humanoid.dbc")
endif()

install(
  DIRECTORY config/
  DESTINATION share/can/config
)

# NOTE: this method of installation is not secure for production env
install(
  FILES ../dbc/humanoid.dbc
  DESTINATION share/can/config
)

ament_package()
