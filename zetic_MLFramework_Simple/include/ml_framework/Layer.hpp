#pragma once
#include <vector>


class Layer {
public:
    virtual std::vector<double>& getWeights() = 0;  // Ensure all weight-managing layers implement this

    virtual ~Layer() {}
    virtual void forward(const std::vector<double>& input, std::vector<double>& output) = 0;
    virtual void backward(const std::vector<double>& input, const std::vector<double>& output,
                          const std::vector<double>& gradOutput, std::vector<double>& gradInput,
                          std::vector<double>& gradWeights) = 0;  // Add gradient w.r.t weights if necessary
};