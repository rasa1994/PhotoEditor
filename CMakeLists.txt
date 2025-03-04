cmake_minimum_required(VERSION 3.29)

set(VCPKG_ROOT "VCPKG_ROOT" CACHE STRING "VCPKG_ROOT")

if (NOT DEFINED VCPKG_ROOT)
  set(VCPKG_ROOT "C:/vcpkg")
endif()

set(CMAKE_TOOLCHAIN_FILE "${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")

project("PhotoEditor" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include ("cmake/ImageHandler.cmake")

add_executable(Main App/Main.cpp)

find_package(wxWidgets CONFIG REQUIRED)
target_link_libraries(Main PUBLIC wx::core wx::base imagehandler)
target_link_directories(Main PUBLIC ${CMAKE_SOURCE_DIR} ${imagehandler_SOURCE_DIR})
target_include_directories(Main PUBLIC ${CMAKE_SOURCE_DIR} ${imagehandler_SOURCE_DIR})
target_sources(Main PRIVATE App/Main.cpp App/Frame.hpp App/App.hpp App/Resources.ixx App/Helper.ixx Filters/Filters.ixx)
