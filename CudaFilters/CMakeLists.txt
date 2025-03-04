add_library(cudalib STATIC FiltersCuda.cu)
target_include_directories(cudalib PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(cudalib PUBLIC cudart)
