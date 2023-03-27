#pragma once
#include <vector>
#include <tuple>
#include "../ActivationFunction/ActivationFunction.h"
#include "../LossFunction/LossFunction.h"

#define N_ITERATIONS 10000
#define LEARNING_RATE 0.01
#define ACTIVATION_FUNCTION "sigmoid"
#define LOSS_FUNCTION "squared_loss"

class Perceptron {
public:
    Perceptron(
        const int& nIterations = N_ITERATIONS,
        const double& learningRate = LEARNING_RATE,
        const std::string& activationFunction = ACTIVATION_FUNCTION,
        const std::string& lossFunction = LOSS_FUNCTION
    );
    ~Perceptron();
    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y);
    std::vector<double> predict(const std::vector<std::vector<double>>& X);
    std::vector<double> predict_proba(const std::vector<std::vector<double>>& X);

    int getNIterations();
    double getLearningRate();
    std::string getActivationFunction();
    std::string getLossFunction();
    std::tuple<int, double, std::string, std::string> getParameters();

    std::vector<double> getWeights();
    double getBias();

private:
    double _predictSingle(const std::vector<double>& X);
    ActivationFunction<double>* _getActivationFunction(const std::string& activationFunction);
    LossFunction<double>* _getLossFunction(const std::string& lossFunction);

    int _nIterations;
    double _learningRate;
    ActivationFunction<double>* _activationFunction;
    LossFunction<double>* _lossFunction;
    std::vector<double> _weights;
    double _bias;
};
