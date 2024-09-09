/*
SoftmaxLayer.hpp
*/ 
#pragma once
#include "Layer.hpp"
#include <cmath>
#include <vector>
#include <iostream>

class SoftmaxLayer : public Layer {
public:
    std::vector<double>& getWeights() override {
        static std::vector<double> dummy;  // Softmax has no weights
        return dummy;
    }

    std::vector<double>& getGradWeights() override {
        static std::vector<double> dummy;  // Softmax has no weight gradients
        return dummy;
    }

    void forward(const std::vector<double>& input, std::vector<double>& output) override {
        double sum = 0.0;
        output.resize(input.size());
        for (double val : input) {
            sum += std::exp(val);
        }
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = std::exp(input[i]) / sum;
        }
        output_ = output;  // Store output for backward pass
    }

    void backward(const std::vector<double>& input, const std::vector<double>& output,
                  const std::vector<double>& gradOutput, std::vector<double>& gradInput,
                  std::vector<double>& gradWeights) override {
        gradInput.resize(output.size());
        for (size_t i = 0; i < output.size(); ++i) {
            gradInput[i] = output[i] - gradOutput[i];  // Simplified gradient for softmax
        }
    }
};