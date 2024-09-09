# Assignment Overview
- Problem Statement:  _This project is a simple implementation of the neural network inference framework described in the assignment. The framework includes the core components necessary to construct and train basic neural networks. My intent was to design an easy-to-use neural network interface that can be extended and applied to various datasets, with the MNIST classification task serving as an example._

- NOTE: _While I aimed to provide a more comprehensive implementation, I was unable to fully develop all the features I had envisioned due to time constraints. The current implementation demonstrates the basics of layer stacking, forward propagation, and backpropagation, but some aspects (e.g., more complex optimizers, additional layer types, or advanced model configurations) have not yet been fully realized._

- _Total time spent on this assignment was approximately 10 hours, and this represents a simpler version of what I had initially planned._
- There are two versions of the assignment:
   - Simple Version: A basic neural network framework showcasing the core functionality such as layer stacking, activation functions, and optimization using a simple binary classification problem (XOR).

   - MNIST Version: A more advanced implementation that includes handling the MNIST dataset, multi-class classification using the Softmax layer, and Cross-Entropy loss for evaluation. This version demonstrates the framework's extension to real-world data.

## Zetic ML Framework - Simple Version

### Overview

The **Zetic ML Framework (Simple Version)** is a lightweight neural network framework implemented in C++. This version is designed for basic binary classification tasks, demonstrating the fundamentals of a feedforward neural network architecture with backpropagation, activation functions, and optimization using Stochastic Gradient Descent (SGD). 

The framework is modular, allowing users to define and stack different layers, specify loss functions, and train a network on basic datasets.

### Features

- **Fully Connected Layers**: Supports fully connected (dense) layers where each neuron is connected to every neuron in the previous layer.
- **Activation Functions**: Includes ReLU and Sigmoid activation functions.
- **Loss Functions**: Supports Binary Cross-Entropy loss for binary classification.
- **Optimization**: Stochastic Gradient Descent (SGD) with gradient clipping to avoid exploding gradients.

### Files Overview

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

### Setup and Compilation

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

### Usage Example

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

### Extending the Framework

- **Adding More Layers**: You can define and add more layers by extending the `Layer` class.
- **Custom Loss Functions**: Implement custom loss functions by inheriting from the `Loss` class.
- **Other Optimizers**: Add new optimizers by extending the `Optimizer` base class.


## Zetic ML Framework - MNIST Version

### Overview

The **Zetic ML Framework (MNIST Version)** extends the Simple version to handle more complex datasets, specifically the MNIST handwritten digit classification dataset. This version showcases how to train and evaluate a neural network using real-world image data, with additional components like the Softmax layer for multi-class classification and Cross-Entropy loss for computing the model's accuracy.

### Features

- **Fully Connected Layers**: Supports dense layers for transforming the input data.
- **Activation Functions**: Includes ReLU, Sigmoid, and Softmax activation functions.
- **Loss Functions**: Implements Cross-Entropy loss for multi-class classification.
- **Optimization**: Stochastic Gradient Descent (SGD) with learning rate control.
- **Dataset Loading**: Includes functionality to load and preprocess the MNIST dataset in `.idx3-ubyte` format.

### Files Overview

1. **main.cpp**: The entry point of the program where the MNIST dataset is loaded, the neural network is constructed, trained, and evaluated.
2. **CMakeLists.txt**: Configuration file for building the project with CMake.
3. **include/ml_framework**: Contains header files for various components of the framework.
   - `MNISTLoader.hpp`: Utility to load and preprocess the MNIST dataset.
   - `FullyConnectedLayer.hpp`: Defines the fully connected layer.
   - `ReLULayer.hpp`: ReLU activation layer.
   - `SigmoidLayer.hpp`: Sigmoid activation layer.
   - `SoftmaxLayer.hpp`: Softmax layer for multi-class classification.
   - `NeuralNetwork.hpp`: Manages layers and orchestrates forward/backward passes.
   - `Loss.hpp`: Contains Cross-Entropy loss implementation.
   - `Optimizer.hpp`: Defines optimizers (SGD).
4. **src/ml_framework**: Source files for the framework's implementation.
   - `FullyConnectedLayer.cpp`: Implementation of the fully connected layer.
   - `NeuralNetwork.cpp`: Implementation of the neural network.
5. **dataset/**: Contains the MNIST dataset files in `.idx3-ubyte` format.
6. **build.sh**: Shell script for building and running the project.

### Setup and Compilation

1. Clone the repository or download the source files.
2. Ensure you have CMake installed (version 3.10 or above).
3. Place the MNIST dataset files (`train-images.idx3-ubyte`, `train-labels.idx1-ubyte`, `t10k-images.idx3-ubyte`, and `t10k-labels.idx1-ubyte`) in the `dataset` directory.
4. Run the `build.sh` script to configure, build, and execute the application:
   
   ```bash
   ./build.sh
   ```

   This script will:
   - Remove any existing `build` directory.
   - Re-create the `build` directory.
   - Run CMake to configure the project.
   - Compile the project with `make`.
   - If successful, run the resulting application.

### Usage Example

The neural network in `main.cpp` is designed to classify MNIST digits:

```cpp
NeuralNetwork net;

net.addLayer(new FullyConnectedLayer(784, 128));  // Input to first hidden layer
net.addLayer(new ReLULayer());
net.addLayer(new FullyConnectedLayer(128, 10));   // Output layer
net.addLayer(new SoftmaxLayer());                 // Softmax for classification

CrossEntropyLoss loss_fn;
SGD optimizer(0.01);

net.train(...);  // Train on MNIST data
```

After training, the network is evaluated on the test set, and the accuracy is printed.

### Extending the Framework

- **Custom Network Architectures**: You can experiment with deeper networks or different configurations by stacking layers in various ways.
- **Custom Loss Functions**: Implement custom loss functions by extending the `Loss` class.
- **Dataset Flexibility**: You can extend `MNISTLoader` to support other datasets.
