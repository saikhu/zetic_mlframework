
# Zetic ML Framework - Simple Version

## Overview

The **Zetic ML Framework (Simple Version)** is a lightweight neural network framework implemented in C++. This version is designed for basic binary classification tasks, demonstrating the fundamentals of a feedforward neural network architecture with backpropagation, activation functions, and optimization using Stochastic Gradient Descent (SGD). 

The framework is modular, allowing users to define and stack different layers, specify loss functions, and train a network on basic datasets.

## Features

- **Fully Connected Layers**: Supports fully connected (dense) layers where each neuron is connected to every neuron in the previous layer.
- **Activation Functions**: Includes ReLU and Sigmoid activation functions.
- **Loss Functions**: Supports Binary Cross-Entropy loss for binary classification.
- **Optimization**: Stochastic Gradient Descent (SGD) with gradient clipping to avoid exploding gradients.

## Files Overview

1. **main.cpp**: The entry point of the program where the neural network is constructed, trained, and evaluated.
2. **CMakeLists.txt**: Configuration file for building the project with CMake.
3. **include/ml_framework**: Contains header files for various components of the framework.
   - `FullyConnectedLayer.hpp`: Defines the fully connected layer.
   - `Layer.hpp`: Base class for layers in the network.
   - `Loss.hpp`: Contains loss function implementations (Binary Cross-Entropy, MSE).
   - `NeuralNetwork.hpp`: The neural network class, managing layers and training process.
   - `Optimizer.hpp`: Defines optimizers (SGD).
   - `ReLULayer.hpp`: ReLU activation layer.
   - `SigmoidLayer.hpp`: Sigmoid activation layer.
4. **src/ml_framework**: Source files for the framework's implementation.
   - `FullyConnectedLayer.cpp`: Implementation of the fully connected layer.
   - `NeuralNetwork.cpp`: Implementation of the neural network class.
5. **build.sh**: Shell script for building and running the project.

## Setup and Compilation

1. Clone the repository or download the source files.
2. Ensure you have CMake installed (version 3.10 or above).
3. Run the `build.sh` script to configure, build, and execute the application:
   
   ```bash
   ./build.sh
   ```

   This script will:
   - Remove any existing `build` directory.
   - Re-create the `build` directory.
   - Run CMake to configure the project.
   - Compile the project with `make`.
   - If successful, run the resulting application.

## Usage Example

The neural network defined in `main.cpp` trains on a simple XOR dataset:

```cpp
NeuralNetwork nn;
nn.addLayer(new FullyConnectedLayer(2, 3));  // Input to Hidden Layer
nn.addLayer(new ReLULayer());
nn.addLayer(new FullyConnectedLayer(3, 1));  // Hidden Layer to Output
nn.addLayer(new SigmoidLayer());             // Sigmoid for output

std::vector<std::vector<double>> inputs = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
std::vector<std::vector<double>> targets = {{0.0}, {1.0}, {1.0}, {0.0}};

BinaryCrossEntropyLoss bce;
SGD optimizer(0.00001);

nn.train(inputs, targets, bce, optimizer);
```

The network is trained for 10,000 epochs and the model output is displayed for each input after training.

## Extending the Framework

- **Adding More Layers**: You can define and add more layers by extending the `Layer` class.
- **Custom Loss Functions**: Implement custom loss functions by inheriting from the `Loss` class.
- **Other Optimizers**: Add new optimizers by extending the `Optimizer` base class.

