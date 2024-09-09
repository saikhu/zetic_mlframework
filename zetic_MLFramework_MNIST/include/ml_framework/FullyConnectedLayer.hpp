/*
FullyConnectedLayer.hpp
*/ 
#pragma once
#include "Layer.hpp"
#include <vector>

class FullyConnectedLayer : public Layer {
public:
    FullyConnectedLayer(int inputSize, int outputSize);
    virtual ~FullyConnectedLayer() {}

    // Override the virtual function from the Layer base class
    void forward(const std::vector<double>& input, std::vector<double>& output) override;

    void backward(const std::vector<double>& input, 
              const std::vector<double>& output,
              const std::vector<double>& gradOutput, 
              std::vector<double>& gradInput,
              std::vector<double>& gradWeights) override;

    std::vector<double>& getWeights() override {
        return weights;
    }

    std::vector<double>& getGradWeights() override {
        return gradWeights_;
    }

private:
    int inputSize;
    int outputSize;
    std::vector<double> weights;
    std::vector<double> biases;
    std::vector<double> gradWeights_;  // To store gradients for the weights
};

