cmake_minimum_required(VERSION 3.8)
project(kcg_moveit_collision_audit)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(geometric_shapes REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(yaml_cpp_vendor REQUIRED)

add_executable(
  connector_pick_collision_audit
  src/connector_pick_collision_audit.cpp
)
add_executable(
  d38999_closure_clearance_analyzer
  src/d38999_closure_clearance_analyzer.cpp
)
add_executable(
  d38999_nut_regrasp_search
  src/d38999_nut_regrasp_search.cpp
)
ament_target_dependencies(
  connector_pick_collision_audit
  geometric_shapes
  moveit_core
  moveit_ros_planning
)
target_link_libraries(
  connector_pick_collision_audit
  OpenSSL::Crypto
  yaml-cpp
)
ament_target_dependencies(
  d38999_closure_clearance_analyzer
  geometric_shapes
  moveit_core
  moveit_ros_planning
)
target_link_libraries(
  d38999_closure_clearance_analyzer
  OpenSSL::Crypto
  yaml-cpp
)
ament_target_dependencies(
  d38999_nut_regrasp_search
  geometric_shapes
  moveit_core
  moveit_ros_planning
)
target_link_libraries(
  d38999_nut_regrasp_search
  OpenSSL::Crypto
  yaml-cpp
)
target_compile_options(
  connector_pick_collision_audit
  PRIVATE -Wall -Wextra -Wpedantic
)
target_compile_options(
  d38999_closure_clearance_analyzer
  PRIVATE -Wall -Wextra -Wpedantic
)
target_compile_options(
  d38999_nut_regrasp_search
  PRIVATE -Wall -Wextra -Wpedantic
)

install(
  TARGETS
    connector_pick_collision_audit
    d38999_closure_clearance_analyzer
    d38999_nut_regrasp_search
  DESTINATION lib/${PROJECT_NAME}
)
install(
  DIRECTORY config docs
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(
    test_collision_audit_contract
    test/test_collision_audit_contract.py
  )
  ament_add_pytest_test(
    test_d38999_nut_regrasp_search_contract
    test/test_d38999_nut_regrasp_search_contract.py
  )
endif()

ament_package()
