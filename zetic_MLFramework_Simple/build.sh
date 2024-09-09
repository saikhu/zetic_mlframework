#!/bin/bash

# Check if the build directory exists
if [ -d "build" ]; then
    # Remove the build directory if it exists
    rm -rf build
fi

# Create the build directory
mkdir build

# Navigate to the build directory
cd build

# Run CMake to configure the project
if cmake ..; then
    echo "CMake configuration successful."
else
    echo "CMake configuration failed."
    exit 1  # Exit the script if CMake fails
fi

# Compile the project
if make; then
    echo "Build successful."
else
    echo "Build failed."
    exit 1  # Exit the script if make fails
fi

# Go back to the root directory
cd ..

# Run the compiled executable
./build/ZeticApp
