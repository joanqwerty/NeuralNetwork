#pragma once

#include <iostream>
#include <fstream>
#include "NeuralNetwork.h"


class App
{
private:
    // Datos de entrada (XOR)
    arma::mat X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    arma::mat y = {0, 1, 1, 0};

    int inputNeurons = 2;
    int hiddenNeurons = 3;
    int outputNeurons = 1;
    int epochs = 100000;
    double learningRate = 0.1;

    NeuralNetwork nn;
    std::string modelFile = "model_xor.json";

public:
    App();
    void start();
    void newTrain();
    void train();
    void predict();
};
