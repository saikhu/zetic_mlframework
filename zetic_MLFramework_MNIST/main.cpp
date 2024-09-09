/*
main.cpp
*/
#include <iostream>
#include <vector>
#include "ml_framework/MNISTLoader.hpp"
#include "ml_framework/NeuralNetwork.hpp"
#include "ml_framework/FullyConnectedLayer.hpp"
#include "ml_framework/ReLULayer.hpp"
#include "ml_framework/SigmoidLayer.hpp"
#include "ml_framework/SoftmaxLayer.hpp" // Include this header
#include "ml_framework/Loss.hpp"
#include "ml_framework/Optimizer.hpp"




std::vector<double> float_to_double(const std::vector<float>& input) {
    return std::vector<double>(input.begin(), input.end());
}

int main() {
    // Load MNIST images and labels
    std::vector<std::vector<float>> images = load_mnist_images("../dataset/train-images.idx3-ubyte");
    std::vector<int> labels = load_mnist_labels("../dataset/train-labels.idx1-ubyte");

    NeuralNetwork net;

    // Define a simple network for MNIST classification
    net.addLayer(new FullyConnectedLayer(784, 128));  // First hidden layer
    net.addLayer(new ReLULayer());
    net.addLayer(new FullyConnectedLayer(128, 10));   // Output layer
    net.addLayer(new SoftmaxLayer());                 // Softmax for classification

    CrossEntropyLoss loss_fn;
    SGD optimizer(0.01);

    int num_epochs = 1;
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float total_loss = 0;
        for (size_t i = 0; i < images.size(); ++i) {
            std::vector<double> prediction;
            net.forward(float_to_double(images[i]), prediction);  // Convert float to double

            std::vector<double> target(10, 0.0);
            target[labels[i]] = 1.0;  // One-hot encode the target

            float loss = loss_fn.compute(prediction, target);
            total_loss += loss;
            std::vector<double> grad = loss_fn.gradLoss(prediction, target);
            
            net.backward(grad);  // Implement backward function
            
            // Update the weights of each layer
            for (auto& layer : net.getLayers()) {
                optimizer.update(layer->getWeights(), layer->getGradWeights());  // Pass the layer's weights and their gradients
            }
        }
        std::cout << "Epoch " << epoch << ", Loss: " << total_loss / images.size() << std::endl;
    }

    // Evaluate on test set
    std::vector<std::vector<float>> test_images = load_mnist_images("../dataset/t10k-images.idx3-ubyte");
    std::vector<int> test_labels = load_mnist_labels("../dataset/t10k-labels.idx1-ubyte");

    int correct = 0;
    for (size_t i = 0; i < test_images.size(); ++i) {
        std::vector<double> prediction;
        net.forward(float_to_double(test_images[i]), prediction);
        int predicted_label = std::distance(prediction.begin(), std::max_element(prediction.begin(), prediction.end()));
        if (predicted_label == test_labels[i]) {
            correct++;
        }
    }
    float accuracy = static_cast<float>(correct) / test_images.size();
    std::cout << "Test Accuracy: " << accuracy << std::endl;

    return 0;
}
