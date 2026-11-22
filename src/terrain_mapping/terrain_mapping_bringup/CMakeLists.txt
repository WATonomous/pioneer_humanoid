cmake_minimum_required(VERSION 3.8)
project(terrain_mapping_bringup)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch config rviz foxglove DESTINATION share/${PROJECT_NAME})
install(PROGRAMS scripts/check_map.py scripts/latency_logger.py DESTINATION lib/${PROJECT_NAME})

ament_package()
