cmake_minimum_required(VERSION 3.8)
project(iiwa_description)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch meshes urdf
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
)

ament_package()
