/*
ReLULayer.hpp
*/
#pragma once
#include "Layer.hpp"
#include <vector>
#include <algorithm>

class ReLULayer : public Layer {
public:
    void forward(const std::vector<double>& input, std::vector<double>& output) override {
        output.resize(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = std::max(0.0, input[i]);
        }
        output_ = output;  // Store output for use in the backward pass
    }

    void backward(const std::vector<double>& input, const std::vector<double>& output,
                  const std::vector<double>& gradOutput, std::vector<double>& gradInput,
                  std::vector<double>& gradWeights) override {
        gradInput.resize(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            gradInput[i] = (input[i] > 0) ? gradOutput[i] : 0.0;  // Derivative of ReLU
        }
    }

    std::vector<double>& getWeights() override {
        static std::vector<double> dummy;  // ReLU has no weights
        return dummy;
    }

    std::vector<double>& getGradWeights() override {
        static std::vector<double> dummy;  // ReLU has no weight gradients
        return dummy;
    }
};
