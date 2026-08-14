cmake_minimum_required(VERSION 3.8)
project(kcg_moveit1)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY config launch worlds
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
)

install(PROGRAMS
  scripts/check_physics_stability.py
  scripts/moveit_smoke_test.py
  DESTINATION lib/${PROJECT_NAME}
)

install(FILES .setup_assistant DESTINATION share/${PROJECT_NAME})

ament_package()
