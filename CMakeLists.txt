cmake_minimum_required(VERSION 3.10)
project(NN)

set(CMAKE_CXX_STANDARD 14)

# Buscar la biblioteca Armadillo
find_package(Armadillo REQUIRED)

# Agregar la carpeta include
include_directories(${CMAKE_SOURCE_DIR}/include)

file(GLOB SOURCES "src/*.cpp")
add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} armadillo)
