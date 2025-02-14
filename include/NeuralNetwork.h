#pragma once

#include <iostream>
#include <armadillo>
#include <fstream>
#include "json.hpp"

using json = nlohmann::json;

class NeuralNetwork
{
private:
    arma::mat W1, W2, b1, b2, A1, A2;
    double learningRate;
    int inputNeurons, hiddenNeurons, outputNeurons;
    void initWeights();

public:
    NeuralNetwork(int inputNeurons, int hiddenNeurons, int outputNeurons, double learningRate);

    void forward(const arma::mat& X);
    void backpropagation(const arma::mat& X, const arma::mat& y);
    void train(const arma::mat& X, const arma::mat& y, int epochs);
    arma::mat predict(const arma::mat& X);
    void setLearningRate(double learningRate);
    void setHiddenNeurons(int hiddenNeurons);


    json matrixToJson(const arma::mat& mat);
    arma::mat jsonToMatrix(const json& j);
    void saveModel(const std::string& filename);
    bool loadModel(const std::string& filename);
    void deleteModel(const std::string& fiilename);
};
