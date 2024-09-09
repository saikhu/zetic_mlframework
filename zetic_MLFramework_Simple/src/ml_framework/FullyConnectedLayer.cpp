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
    double limit = std::sqrt(1.0 / (inputSize));  // Smaller limit for weights
    std::uniform_real_distribution<> dis(-limit, limit);

    for (auto& weight : weights) {
        weight = dis(gen);
    }
    for (auto& bias : biases) {
        bias = 0;  // Initialize biases to zero
    }
}


void FullyConnectedLayer::forward(const std::vector<double>& input, std::vector<double>& output) {
    
    output.resize(outputSize, 0.0);

    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
            output[i] += input[j] * weights[i * inputSize + j];
        }
        output[i] += biases[i];
    }
//     std::cout << "FullyConnectedLayer::forward - Output: ";
//     for (const auto& o : output) std::cout << o << " ";
//     std::cout << std::endl;
}


void FullyConnectedLayer::backward(const std::vector<double>& input, const std::vector<double>& output,
                                   const std::vector<double>& gradOutput, std::vector<double>& gradInput,
                                   std::vector<double>& gradWeights) {
    // Set a clip value to prevent exploding gradients
    double clipValue = 1.0;  // Adjust this as necessary

    // Ensure the sizes match to avoid runtime errors
    if (output.size() != gradOutput.size()) {
        std::cerr << "Error: Output size and gradOutput size mismatch in FullyConnectedLayer." << std::endl;
        return;
    }

    // Initialize gradients
    gradInput.assign(input.size(), 0.0);  // Gradient w.r.t. input
    gradWeights.assign(weights.size(), 0.0);  // Gradient w.r.t. weights

    // Compute gradients for input and weights
    for (size_t i = 0; i < outputSize; ++i) {
        if (gradOutput[i] == 0.0) continue; // Skip zero gradients
        for (size_t j = 0; j < inputSize; ++j) {
            // Gradient for the input
            gradInput[j] += weights[i * inputSize + j] * gradOutput[i];

            // Gradient for the weights
            gradWeights[i * inputSize + j] = input[j] * gradOutput[i];

            // Apply gradient clipping to prevent exploding gradients
            if (gradWeights[i * inputSize + j] > clipValue) {
                gradWeights[i * inputSize + j] = clipValue;
            } else if (gradWeights[i * inputSize + j] < -clipValue) {
                gradWeights[i * inputSize + j] = -clipValue;
            }
        }
        // Gradient for biases
        biases[i] += gradOutput[i];

        // Clip bias updates
        if (biases[i] > clipValue) {
            biases[i] = clipValue;
        } else if (biases[i] < -clipValue) {
            biases[i] = -clipValue;
        }
    }

    // Log gradients for debugging purposes
    // std::cout << "FullyConnectedLayer::backward - gradInput: ";
    // for (const auto& gi : gradInput) std::cout << gi << " ";
    // std::cout << std::endl;

    // std::cout << "FullyConnectedLayer::backward - gradWeights: ";
    // for (const auto& gw : gradWeights) std::cout << gw << " ";
    // std::cout << std::endl;

    // std::cout << "FullyConnectedLayer::backward - biases: ";
    // for (const auto& b : biases) std::cout << b << " ";
    // std::cout << std::endl;
}


