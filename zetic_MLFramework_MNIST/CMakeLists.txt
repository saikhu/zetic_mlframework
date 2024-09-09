cmake_minimum_required(VERSION 3.10)
project(ZeticMLFramework)

set(CMAKE_CXX_STANDARD 17)  # Set the C++ standard
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include the headers
include_directories(include)

# Add the source directory
add_subdirectory(src)

# Add executable
add_executable(ZeticApp main.cpp)

# Link the executable with the ML framework library
target_link_libraries(ZeticApp MLFramework)
