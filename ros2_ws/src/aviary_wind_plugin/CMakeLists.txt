cmake_minimum_required(VERSION 3.8)
project(aviary_wind_plugin)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)

# Use PkgConfig to bypass the broken Gazebo CMake setup in this Docker container
find_package(PkgConfig REQUIRED)
pkg_check_modules(GZ_SIM REQUIRED gz-sim8)
pkg_check_modules(GZ_PLUGIN REQUIRED gz-plugin2)
pkg_check_modules(GZ_MATH REQUIRED gz-math7)

add_library(LocalizedWindPlugin SHARED src/LocalizedWindPlugin.cc)
target_compile_features(LocalizedWindPlugin PUBLIC cxx_std_17)

# Manually map the directories and libraries found by PkgConfig
target_include_directories(LocalizedWindPlugin BEFORE PRIVATE
  src
  ${GZ_SIM_INCLUDE_DIRS}
  ${GZ_PLUGIN_INCLUDE_DIRS}
  ${GZ_MATH_INCLUDE_DIRS}
)

target_link_directories(LocalizedWindPlugin PRIVATE
  ${GZ_SIM_LIBRARY_DIRS}
  ${GZ_PLUGIN_LIBRARY_DIRS}
  ${GZ_MATH_LIBRARY_DIRS}
)

target_link_libraries(LocalizedWindPlugin
  ${GZ_SIM_LIBRARIES}
  ${GZ_PLUGIN_LIBRARIES}
  ${GZ_MATH_LIBRARIES}
)

install(TARGETS LocalizedWindPlugin
  DESTINATION lib
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()