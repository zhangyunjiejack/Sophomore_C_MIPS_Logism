cmake_minimum_required(VERSION 3.6)
project(proj1_ait)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        hashtable.c
        hashtable.h
        Makefile
        philspel.c
        philspel.h
        sampleDictionary
        sampleInput
        sampleOutput
        testOutput)

add_executable(proj1_ait ${SOURCE_FILES})