/*
Loss.hpp
*/
#pragma once
#include <vector>
#include <cmath>

class Loss {
public:
    virtual ~Loss() = default;
    virtual double compute(const std::vector<double>& predicted, const std::vector<double>& actual) = 0;
    virtual std::vector<double> gradLoss(const std::vector<double>& predicted, const std::vector<double>& actual) = 0;
};

class CrossEntropyLoss : public Loss {
public:
    double compute(const std::vector<double>& output, const std::vector<double>& target) override {
        double loss = 0.0;
        for (size_t i = 0; i < output.size(); ++i) {
            loss -= target[i] * std::log(output[i] + 1e-9); // Adding a small value to avoid log(0)
        }
        return loss;
    }

    std::vector<double> gradLoss(const std::vector<double>& output, const std::vector<double>& target) override {
        std::vector<double> grad(output.size());
        for (size_t i = 0; i < output.size(); ++i) {
            grad[i] = output[i] - target[i]; // Gradient of cross-entropy
        }
        return grad;
    }
};

class MSELoss : public Loss {
public:
    double compute(const std::vector<double>& predicted, const std::vector<double>& actual) override {
        double sum = 0.0;
        for (size_t i = 0; i < predicted.size(); ++i) {
            double diff = predicted[i] - actual[i];
            sum += diff * diff;
        }
        return sum / predicted.size();
    }

    std::vector<double> gradLoss(const std::vector<double>& predicted, const std::vector<double>& actual) override {
        std::vector<double> grad(predicted.size());
        for (size_t i = 0; i < predicted.size(); ++i) {
            grad[i] = 2 * (predicted[i] - actual[i]);
        }
        return grad;
    }
};

class BinaryCrossEntropyLoss : public Loss {
public:
    double epsilon = 1e-12;
    double compute(const std::vector<double>& output, const std::vector<double>& target) override {
        double loss = 0.0;
        for (size_t i = 0; i < output.size(); ++i) {
            double o = std::max(epsilon, std::min(1.0 - epsilon, output[i]));  // Clamp output
            loss += -(target[i] * log(o) + (1 - target[i]) * log(1 - o));
        }
        return loss / output.size();
    }

    std::vector<double> gradLoss(const std::vector<double>& output, const std::vector<double>& target) override {
        std::vector<double> grad(output.size());
        for (size_t i = 0; i < output.size(); ++i) {
            grad[i] = (output[i] - target[i]) / (output[i] * (1 - output[i]));
        }
        return grad;
    }
};

