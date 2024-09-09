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
              std::vector<double>& gradWeights);


    std::vector<double>& getWeights() override {
        return weights;
    }

private:
    int inputSize;
    int outputSize;
    std::vector<double> weights;
    std::vector<double> biases;
};
