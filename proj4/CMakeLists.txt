cmake_minimum_required(VERSION 3.7)
project(proj4_ait)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        src/cnn.c
        src/main.c
        src/python.c
        src/timestamp.c
        src/util.c)

add_executable(proj4_ait ${SOURCE_FILES})