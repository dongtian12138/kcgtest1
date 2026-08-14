cmake_minimum_required(VERSION 3.8)
project(kcg_grasping)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY config launch worlds
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
)

install(FILES README.md DESTINATION share/${PROJECT_NAME})

install(PROGRAMS
  scripts/cylinder_task_manager.py
  scripts/scripted_cylinder_grasp.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
