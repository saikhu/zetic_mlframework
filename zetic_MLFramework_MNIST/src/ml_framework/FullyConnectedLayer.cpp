/*
FullyConnectedLayer.cpp
*/ 
#include "ml_framework/FullyConnectedLayer.hpp"
#include <random>
#include <iostream>

#include <cmath>

FullyConnectedLayer::FullyConnectedLayer(int inputSize, int outputSize)
    : inputSize(inputSize), outputSize(outputSize), weights(inputSize * outputSize), biases(outputSize) {
    
    std::random_device rd;
    std::mt19937 gen(rd());
    double limit = std::sqrt(2.0 / (inputSize));  // Using He initialization for ReLU layers
    std::uniform_real_distribution<> dis(-limit, limit);

    std::cout << "Initializing Weights and Biases..." << std::endl;
    
    for (auto& weight : weights) {
        weight = dis(gen);
        std::cout << weight << " ";  // Log each weight
    }
    std::cout << std::endl;

    for (auto& bias : biases) {
        bias = 0;  // Initialize biases to zero
    }
    std::cout << "Biases initialized to 0" << std::endl;
}

void FullyConnectedLayer::forward(const std::vector<double>& input, std::vector<double>& output) {
    output_.resize(outputSize);  // Ensure the output size is consistent
    if (input.size() != inputSize) {
        std::cerr << "Input size mismatch in FullyConnectedLayer: expected " << inputSize << ", got " << input.size() << std::endl;
        return;
    }

    for (int i = 0; i < outputSize; ++i) {
        output_[i] = 0;
        for (int j = 0; j < inputSize; ++j) {
            output_[i] += input[j] * weights[i * inputSize + j];
        }
        output_[i] += biases[i];
    }
    output = output_;  // Assign the stored output to the argument output
    
    std::cout << "FullyConnectedLayer::forward - Output: ";
    for (const auto& o : output) std::cout << o << " ";
    std::cout << std::endl;
}

void FullyConnectedLayer::backward(const std::vector<double>& input, const std::vector<double>& output,
                                   const std::vector<double>& gradOutput, std::vector<double>& gradInput,
                                   std::vector<double>& gradWeights) {
    gradInput.assign(input.size(), 0.0);  // Gradient w.r.t. input
    gradWeights.assign(weights.size(), 0.0);  // Gradient w.r.t. weights

    if (output.size() != gradOutput.size()) {
        std::cerr << "Output size and gradOutput size mismatch in FullyConnectedLayer backward" << std::endl;
        return;
    }

    for (size_t i = 0; i < outputSize; ++i) {
        if (gradOutput[i] == 0.0) continue; // Skip zero gradients
        for (size_t j = 0; j < inputSize; ++j) {
            gradInput[j] += weights[i * inputSize + j] * gradOutput[i];
            gradWeights[i * inputSize + j] = input[j] * gradOutput[i];
        }
        biases[i] += gradOutput[i];
    }

    std::cout << "FullyConnectedLayer::backward - gradInput: ";
    for (const auto& gi : gradInput) std::cout << gi << " ";
    std::cout << std::endl;

    std::cout << "FullyConnectedLayer::backward - gradWeights: ";
    for (const auto& gw : gradWeights) std::cout << gw << " ";
    std::cout << std::endl;

    std::cout << "FullyConnectedLayer::backward - biases: ";
    for (const auto& b : biases) std::cout << b << " ";
    std::cout << std::endl;
}



