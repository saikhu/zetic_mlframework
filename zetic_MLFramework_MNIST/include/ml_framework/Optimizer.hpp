/*
Optimizer.hpp
*/

#pragma once
#include <vector>
#include <iostream>

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void update(std::vector<double>& weights, const std::vector<double>& gradWeights) = 0;
};

class SGD : public Optimizer {
public:
    SGD(double lr, double clip_value = 10.0) : learningRate(lr), clipValue(clip_value) {}

    void update(std::vector<double>& weights, const std::vector<double>& gradWeights) override {
        for (size_t i = 0; i < weights.size(); ++i) {
            double gradient = gradWeights[i];
            gradient = std::max(-clipValue, std::min(clipValue, gradient));  // Clip gradients
            weights[i] -= learningRate * gradient;
        }

        std::cout << "SGD::update - Weights: ";
        for (const auto& w : weights) std::cout << w << " ";
        std::cout << std::endl;
    }

private:
    double learningRate;
    double clipValue;
};
