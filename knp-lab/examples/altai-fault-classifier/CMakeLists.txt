cmake_minimum_required(VERSION 3.25)
project(altai-fault-classifier)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT TARGET KNP::BaseFramework::Core)
    find_package(knp-base-framework REQUIRED)
endif()

add_executable("${PROJECT_NAME}" main.cpp)
target_link_libraries("${PROJECT_NAME}" PRIVATE KNP::BaseFramework::Core)
