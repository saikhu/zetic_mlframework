/*
main.cpp
*/
#include <iostream>
#include <vector>
#include "ml_framework/NeuralNetwork.hpp"
#include "ml_framework/FullyConnectedLayer.hpp"
#include "ml_framework/ReLULayer.hpp"
#include "ml_framework/SigmoidLayer.hpp"
#include "ml_framework/Loss.hpp"
#include "ml_framework/Optimizer.hpp"
int main() {
    NeuralNetwork nn;
    nn.addLayer(new FullyConnectedLayer(2, 3));  // Input to Hidden Layer
    nn.addLayer(new ReLULayer());
    nn.addLayer(new FullyConnectedLayer(3, 1));  // Hidden Layer to Output
    nn.addLayer(new SigmoidLayer());             // Sigmoid for output

    std::vector<std::vector<double>> inputs = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
    std::vector<std::vector<double>> targets = {{0.0}, {1.0}, {1.0}, {0.0}};

    BinaryCrossEntropyLoss bce;
    SGD optimizer(0.00001);  // Reduced learning rate

    int epochs = 10000;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            std::vector<double> output;  // Declare output for each input
            nn.forward(inputs[i], output);  // Forward pass
            total_loss += bce.compute(output, targets[i]);
            nn.train(inputs[i], targets[i], bce, optimizer);  // Train the network
        }

        if (epoch % 1000 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << total_loss / inputs.size() << std::endl;
        }
    }

    // 5. Evaluate the trained model
    std::cout << "Trained Outputs:" << std::endl;
    for (auto& input : inputs) {
        std::vector<double> output;
        nn.forward(input, output);
        std::cout << "Input: (" << input[0] << ", " << input[1] << ") - Output: " << output[0] << std::endl;
    }

    return 0;
}