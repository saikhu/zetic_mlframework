/*
ReLULayer.hpp
*/
#pragma once
#include "Layer.hpp"
#include <algorithm>
#include <vector>
#include <iostream>

class ReLULayer : public Layer {
public:
    void forward(const std::vector<double>& input, std::vector<double>& output) override {
        output.resize(input.size());
        std::transform(input.begin(), input.end(), output.begin(), [](double x) {
            return std::max(0.0, x);
        });

        // std::cout << "ReLULayer::forward - Output: ";
        // for (const auto& o : output) std::cout << o << " ";
        // std::cout << std::endl;
    }

    std::vector<double>& getWeights() override {
        static std::vector<double> dummy;  // Empty, as ReLU does not have weights
        return dummy;
    }

    void backward(const std::vector<double>& input, const std::vector<double>& output,
                         const std::vector<double>& gradOutput, std::vector<double>& gradInput,
                         std::vector<double>& gradWeights) override {


        gradInput.resize(input.size());

        for (size_t i = 0; i < input.size(); ++i) {
            gradInput[i] = (input[i] > 0) ? gradOutput[i] : 0.0;
        }

        // std::cout << "ReLULayer::backward - gradInput: ";
        // for (const auto& gi : gradInput) std::cout << gi << " ";
        // std::cout << std::endl;
    }
};
