/*
SigmoidLayer.hpp
*/ 
#pragma once
#include "Layer.hpp"
#include <cmath>
#include <vector>
#include<iostream>

class SigmoidLayer : public Layer {
public:
    void forward(const std::vector<double>& input, std::vector<double>& output) override {
        // std::cout << "SigmoidLayer::forward" << std::endl;
        output.resize(input.size());
        std::transform(input.begin(), input.end(), output.begin(), [](double x) {
            return 1.0 / (1.0 + exp(-x));  // Sigmoid function
        });

        std::cout << "SigmoidLayer::forward - Output: ";
        for (const auto& o : output) std::cout << o << " ";
        std::cout << std::endl;
        // std::cout << "SigmoidLayer::forward: Output size (after resizing): " << output.size() << std::endl;
    }

    std::vector<double>& getWeights() override {
        static std::vector<double> dummy;  // Sigmoid does not have weights
        return dummy;
    }

   void backward(const std::vector<double>& input, const std::vector<double>& output,
                            const std::vector<double>& gradOutput, std::vector<double>& gradInput,
                            std::vector<double>& gradWeights) {
        // std::cout << "SigmoidLayer::backward" << std::endl;
        // std::cout << "Input size: " << input.size() << ", Output size: " << output.size()
        //         << ", gradOutput size: " << gradOutput.size() << std::endl;

        gradInput.resize(input.size());

        if (input.size() != output.size() || output.size() != gradOutput.size()) {
            std::cerr << "Error: Vector size mismatch in backward computation of Sigmoid Layer." << std::endl;
            return;
        }

        gradInput.resize(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            double sigmoid = output[i];
            gradInput[i] = gradOutput[i] * sigmoid * (1 - sigmoid);  // Sigmoid gradient
        }

        std::cout << "SigmoidLayer::backward - gradInput: ";
        for (const auto& gi : gradInput) std::cout << gi << " ";
        std::cout << std::endl;

        // std::cout << "SigmoidLayer::backward: GradInput size: " << gradInput.size() << std::endl;
    }

};
