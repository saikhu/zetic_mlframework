/*
Layer.hpp
*/
#pragma once
#include <vector>

class Layer {
public:
    virtual std::vector<double>& getWeights() = 0;  // Ensure all weight-managing layers implement this
    virtual std::vector<double>& getGradWeights() = 0;  // To get the gradient of the weights

    virtual ~Layer() {}
    
    // Forward and backward functions
    virtual void forward(const std::vector<double>& input, std::vector<double>& output) = 0;
    virtual void backward(const std::vector<double>& input, 
                          const std::vector<double>& output,
                          const std::vector<double>& gradOutput, 
                          std::vector<double>& gradInput, 
                          std::vector<double>& gradWeights) = 0;

    // Method to return the stored output after the forward pass
    const std::vector<double>& getOutput() const { return output_; }

protected:
    std::vector<double> output_;  // Store output of the forward pass
};
