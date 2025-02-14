#include "App.h"

App::App() : nn(inputNeurons, hiddenNeurons, outputNeurons, learningRate)
{
    y = y.t();
}


void App::start()
{
    std::cout << "Red Neuronal XOR" << std::endl;

    int option;

    do
    {
        std::cout << "\n--- Menu ---\n";
        std::cout << "1. Entrenar el modelo\n";
        std::cout << "2. Hacer una predicción\n";
        std::cout << "3. Salir\n";
        std::cout << "\nSeleccione una opción: ";
        std::cin >> option;

        switch (option)
        {
        case 1:
            train();
            break;
        case 2:
            predict();
            break;
        case 3:
            std::cout << "Saliendo...\n";
            break;
        default:
            std::cout << "Opción no válida, intente de nuevo.\n";
            break;
        }
    }
    while (option != 3);

}

void App::newTrain()
{
    std::cout << "Modelo no encontrado. Entrenando desde cero...\n";

    std::cout << "Ingrese la cantidad de capas ocultas: ";
    std::cin >> hiddenNeurons;
    nn.setHiddenNeurons(hiddenNeurons);

    std::cout << "Ingrese learning rate: ";
    std::cin >> learningRate;
    nn.setLearningRate(learningRate);

    std::cout << "Ingrese la cantidad de epocas: ";
    std::cin >> epochs;

    nn.train(X, y, epochs);
    nn.saveModel(modelFile);
}

void App::train()
{
    if (!nn.loadModel(modelFile))
    {
        newTrain();
        return;
    }

    std::cout << "Modelo preentrenado...\n";

    int option;

    do
    {
        std::cout << "\n--- Menu ---\n";
        std::cout << "1. Continuar\n";
        std::cout << "2. Modificar Learning rate\n";
        std::cout << "3. Modificar número de capas ocultas\n";
        std::cout << "\nSeleccione una opción: ";
        std::cin >> option;

        switch (option)
        {

        case 1:
            break;
        case 2:
            std::cout << "Ingrese learning rate: ";
            std::cin >> learningRate;
            nn.setLearningRate(learningRate);
            break;
        case 3:
            std::cout << "Ingrese la cantidad de capas ocultas: ";
            std::cin >> hiddenNeurons;
            nn.setHiddenNeurons(hiddenNeurons);
            break;
        default:
            std::cout << "Opción no válida, intente de nuevo.\n";
            break;
        }

    }
    while(option < 1 || option > 3);

    std::cout << "Ingrese la cantidad de epocas: ";
    std::cin >> epochs;

    nn.train(X, y, epochs);
    nn.saveModel(modelFile);
}

void App::predict()
{
    if (!nn.loadModel(modelFile))
    {
        std::cout << "Modelo no encontrado...\n";
        return;
    }

    arma::mat output = nn.predict(X);
    std::cout << "Predicciones:\n" << output << std::endl;
}
