/*
NeuralNetwork.hpp
*/ 

#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include "Layer.hpp"
#include "Loss.hpp"
#include "Optimizer.hpp"
#include <vector>
#include <memory>

class NeuralNetwork {
public:
    NeuralNetwork() = default;
    ~NeuralNetwork();

    // Disable copy and move semantics
    NeuralNetwork(const NeuralNetwork&) = delete;
    NeuralNetwork& operator=(const NeuralNetwork&) = delete;
    NeuralNetwork(NeuralNetwork&&) = delete;
    NeuralNetwork& operator=(NeuralNetwork&&) = delete;

    // Function to add layers to the network
    void addLayer(Layer* layer);

    // Function to perform forward propagation through all layers
    void forward(const std::vector<double>& input, std::vector<double>& output);

    void backward(const std::vector<double>& gradOutput);

    // Updated train function: now accepts loss function and optimizer as parameters
    void train(const std::vector<double>& input, const std::vector<double>& target, Loss& loss, Optimizer& optimizer);

    std::vector<std::unique_ptr<Layer>>& getLayers() {
    return layers;
}


private:
    std::vector<std::unique_ptr<Layer>> layers;  // Vector to store layer pointers
};

#endif // NEURAL_NETWORK_HPP
