#include "NeuralNetwork.h"

arma::mat sigmoid(const arma::mat &z)
{
    return 1.0 / (1.0 + exp(-z));
}

arma::mat sigmoid_deriv(const arma::mat &a)
{
    return a % (1 - a);
}


void NeuralNetwork::initWeights()
{
    W1 = arma::randn(inputNeurons, hiddenNeurons) * sqrt(1.0 / inputNeurons);
    W2 = arma::randn(hiddenNeurons, outputNeurons) * sqrt(1.0 / inputNeurons);
    b1 = arma::zeros(1, hiddenNeurons);
    b2 = arma::zeros(1, outputNeurons);
}

NeuralNetwork::NeuralNetwork(int inputNeurons, int hiddenNeurons, int outputNeurons, double learningRate)
    : inputNeurons(inputNeurons), hiddenNeurons(hiddenNeurons),
      outputNeurons(outputNeurons), learningRate(learningRate)
{
    initWeights();
}

void NeuralNetwork::setHiddenNeurons(int hiddenNeurons)
{
    this-> hiddenNeurons = hiddenNeurons;
}

void NeuralNetwork::setLearningRate(double learningRate)
{
    this->learningRate = learningRate;
}

void  NeuralNetwork::forward(const arma::mat& X)
{
    arma::mat Z1 = X * W1 + arma::repmat(b1, X.n_rows, 1);
    A1 = sigmoid(Z1);

    arma::mat Z2 = A1 * W2 + arma::repmat(b2, A1.n_rows, 1);
    A2 = sigmoid(Z2);
}

void NeuralNetwork::backpropagation(const arma::mat& X, const arma::mat& y)
{
    int m = X.n_rows;

    arma::mat d_loss = A2 - y;
    arma::mat dA2 = d_loss % sigmoid_deriv(A2);
    arma::mat dW2 = (A1.t() * dA2) / m;
    arma::mat db2 = arma::sum(dA2, 0) / m;

    arma::mat dA1 = dA2 * W2.t();
    arma::mat dZ1 = dA1 % sigmoid_deriv(A1);
    arma::mat dW1 = (X.t() * dZ1) / m;
    arma::mat db1 = arma::sum(dZ1, 0) / m;

    W1 -= learningRate * dW1;
    W2 -= learningRate * dW2;
    b1 -= learningRate * db1;
    b2 -= learningRate * db2;
}

void NeuralNetwork::train(const arma::mat& X, const arma::mat& y, int epochs)
{
    std::cout << "\n===============================\n";
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        forward(X);
        backpropagation(X, y);

        if (epoch % 100 == 0)
        {
            double loss = arma::accu(arma::pow(A2 - y, 2)) / y.n_rows;
            std::cout << "Epoch " << epoch << " Loss: " << loss << std::endl;
        }
    }
    std::cout << "===============================\n";
}

arma::mat NeuralNetwork::predict(const arma::mat& X)
{
    forward(X);
    return A2.transform([](double val)
    {
        return val >= 0.5 ? 1.0 : 0.0;
    });
}

json NeuralNetwork::matrixToJson(const arma::mat& mat)
{
    json j;
    j["rows"] = mat.n_rows;
    j["cols"] = mat.n_cols;
    j["data"] = std::vector<double>(mat.begin(), mat.end());
    return j;
}

arma::mat NeuralNetwork::jsonToMatrix(const json& j)
{
    arma::mat mat(j["rows"], j["cols"]);
    std::vector<double> data = j["data"];
    std::memcpy(mat.memptr(), data.data(), data.size() * sizeof(double));
    return mat;
}

void NeuralNetwork::saveModel(const std::string& filename)
{
    json model;
    model["W1"] = matrixToJson(W1);
    model["W2"] = matrixToJson(W2);
    model["b1"] = matrixToJson(b1);
    model["b2"] = matrixToJson(b2);

    std::ofstream file(filename);
    file << model.dump(4);  // Guarda con indentación de 4 espacios
    file.close();
    std::cout << "\nModelo guardado en " << filename << std::endl;
}

bool NeuralNetwork::loadModel(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        return false;
    }

    json model;
    file >> model;
    file.close();

    W1 = jsonToMatrix(model["W1"]);
    W2 = jsonToMatrix(model["W2"]);
    b1 = jsonToMatrix(model["b1"]);
    b2 = jsonToMatrix(model["b2"]);

    std::cout << "Modelo cargado desde " << filename << std::endl;
    return true;
}

