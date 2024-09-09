// MNISTLoader.hpp
#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>

std::vector<std::vector<float>> load_mnist_images(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Could not open file");

    int magic_number = 0;
    int number_of_images = 0;
    int rows = 0;
    int cols = 0;

    file.read(reinterpret_cast<char*>(&magic_number), 4);
    magic_number = __builtin_bswap32(magic_number);  // Convert endianness

    file.read(reinterpret_cast<char*>(&number_of_images), 4);
    number_of_images = __builtin_bswap32(number_of_images);

    file.read(reinterpret_cast<char*>(&rows), 4);
    rows = __builtin_bswap32(rows);

    file.read(reinterpret_cast<char*>(&cols), 4);
    cols = __builtin_bswap32(cols);

    std::vector<std::vector<float>> images(number_of_images, std::vector<float>(rows * cols));

    for (int i = 0; i < number_of_images; ++i) {
        for (int j = 0; j < rows * cols; ++j) {
            unsigned char pixel = 0;
            file.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
            images[i][j] = static_cast<float>(pixel) / 255.0f;  // Normalize to [0, 1]
        }
    }

    return images;
}

std::vector<int> load_mnist_labels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Could not open file");

    int magic_number = 0;
    int number_of_labels = 0;

    file.read(reinterpret_cast<char*>(&magic_number), 4);
    magic_number = __builtin_bswap32(magic_number);  // Convert endianness

    file.read(reinterpret_cast<char*>(&number_of_labels), 4);
    number_of_labels = __builtin_bswap32(number_of_labels);

    std::vector<int> labels(number_of_labels);

    for (int i = 0; i < number_of_labels; ++i) {
        unsigned char label = 0;
        file.read(reinterpret_cast<char*>(&label), sizeof(label));
        labels[i] = static_cast<int>(label);
    }

    return labels;
}
