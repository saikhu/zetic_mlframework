# Add ML framework library
add_library(MLFramework
    ml_framework/NeuralNetwork.cpp
    ml_framework/FullyConnectedLayer.cpp  
)

# Include directories for MLFramework library
target_include_directories(MLFramework PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/../include  # Adjust this path as necessary
)