/*
NeuralNetwork.cpp
*/ 
#include "ml_framework/NeuralNetwork.hpp"
#include "ml_framework/Loss.hpp"
#include "ml_framework/Optimizer.hpp"

NeuralNetwork::~NeuralNetwork() {
    // unique_ptr automatically deletes the layers, so no need to manually clean up
}

void NeuralNetwork::addLayer(Layer* layer) {
    layers.emplace_back(layer);  // Take ownership of the layer
}


void NeuralNetwork::forward(const std::vector<double>& input, std::vector<double>& output) {
    // std::cout << "NeuralNetwork::forward" << std::endl;
    std::vector<double> currentInput = input;
    std::vector<double> currentOutput;
    std::vector<std::vector<double>> intermediateOutputs;  // Store outputs for each layer

    for (auto& layer : layers) {
        layer->forward(currentInput, currentOutput);
        currentInput = currentOutput;  // Output becomes next layer's input
        currentOutput.clear();
    }
    output = currentInput;  // Final output
    // std::cout << "NeuralNetwork::forward: Final output size: " << output.size() << std::endl;
}



void NeuralNetwork::train(const std::vector<double>& input, const std::vector<double>& target, Loss& loss, Optimizer& optimizer) {
    // 1. Forward pass
    std::vector<double> output;
    std::vector<std::vector<double>> intermediateOutputs;
    std::vector<double> currentInput = input;

    // Forward pass through layers
    for (auto& layer : layers) {
        std::vector<double> currentOutput;
        layer->forward(currentInput, currentOutput);
        intermediateOutputs.push_back(currentOutput);
        currentInput = currentOutput;
    }
    output = intermediateOutputs.back();  // Final output

    // 2. Compute loss and its gradient
    std::vector<double> gradOutput = loss.gradLoss(output, target);

    // 3. Backward pass through layers
    std::vector<double> gradInput(input.size(), 0);
    for (int i = layers.size() - 1; i >= 0; --i) {  // Backward through layers
        std::vector<double> gradWeights;
        const std::vector<double>& currentLayerInput = (i == 0) ? input : intermediateOutputs[i - 1];
        layers[i]->backward(currentLayerInput, intermediateOutputs[i], gradOutput, gradInput, gradWeights);
        optimizer.update(layers[i]->getWeights(), gradWeights);
        gradOutput = gradInput;  // Pass the gradient back to previous layers
    }
}
