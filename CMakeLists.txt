cmake_minimum_required(VERSION 3.17)
project(MUL)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-Wall -Wextra -pedantic)

include_directories(external/stbi)

file(GLOB_RECURSE HEADERS_FILES "src/*.h")
file(GLOB_RECURSE SOURCE_FILES "src/*.cpp")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")


add_executable(JPEG_encoder ${SOURCE_FILES} ${HEADER_FILES})
target_link_libraries(JPEG_encoder fftw3 fftw3f pthread)
