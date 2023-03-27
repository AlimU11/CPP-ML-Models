#include <random>
#include <algorithm>
#include <numeric>
#include "Perceptron.h"
#include "../ActivationFunction/Sigmoid.h"
#include "../ActivationFunction/SoftMax.h"
#include "../ActivationFunction/ReLU.h"
#include "../ActivationFunction/LeakyReLU.h"
#include "../ActivationFunction/Tanh.h"
#include "../LossFunction/SquaredLoss.h"


Perceptron::Perceptron(
    const int& nIterations,
    const double& learningRate,
    const std::string& activationFunction,
    const std::string& lossFunction) {
    _nIterations = nIterations;
    _learningRate = learningRate;
    _activationFunction = _getActivationFunction(activationFunction);
    _lossFunction = _getLossFunction(lossFunction);
}

Perceptron::~Perceptron() {
    delete _activationFunction;
    delete _lossFunction;
}

void Perceptron::fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    size_t n_samples = X.size();
    size_t n_features = X[0].size();

    _weights = std::vector<double>(n_features);
    _bias = dis(gen);

    for (int i = 0; i < n_features; i++) {
        _weights[i] = dis(gen);
    }

    for (int epoch = 0; epoch < _nIterations; epoch++) {
        std::vector<double> linearOutput(n_samples);
        std::vector<double> y_pred(n_samples);
        std::vector<double> loss(n_samples);
        std::vector<double> derivative = _activationFunction->derivate(linearOutput);

        std::transform(X.begin(), X.end(), linearOutput.begin(), [&](const std::vector<double>& x) { return std::inner_product(x.begin(), x.end(), _weights.begin(), 0.0) + _bias; });

        y_pred = _activationFunction->activate(linearOutput);

        for (int i = 0; i < n_samples; i++) {

            loss[i] = _lossFunction->gradient(y[i], y_pred[i]) * derivative[i];

            std::vector<double> loss_weights(n_features);
            std::transform(X[i].begin(), X[i].end(), loss_weights.begin(), [&](const double& x) { return loss[i] * x; });

            double loss_bias = loss[i];

            std::transform(
                _weights.begin(),
                _weights.end(),
                loss_weights.begin(),
                _weights.begin(),
                [&](const double& w, const double& lw) { return w - _learningRate * lw; });

            _bias -= _learningRate * loss_bias;
        }
    }
}

std::vector<double> Perceptron::predict(const std::vector<std::vector<double>>& X) {
    std::vector<double> y_pred = predict_proba(X);
    std::transform(X.begin(), X.end(), y_pred.begin(), [&](const std::vector<double>& x) { return _predictSingle(x) > 0.5 ? 1 : 0; });

    return y_pred;
}

std::vector<double> Perceptron::predict_proba(const std::vector<std::vector<double>>& X) {
    std::vector<double> y_pred(X.size());
    std::transform(X.begin(), X.end(), y_pred.begin(), [&](const std::vector<double>& x) { return _predictSingle(x); });

    return y_pred;
}

int Perceptron::getNIterations() {
    return _nIterations;
}

double Perceptron::getLearningRate() {
    return _learningRate;
}

std::string Perceptron::getActivationFunction() {
    return _activationFunction->getName();
}

std::string Perceptron::getLossFunction() {
    return _lossFunction->getName();
}

std::tuple<int, double, std::string, std::string> Perceptron::getParameters() {
    return std::make_tuple(
        _nIterations,
        _learningRate,
        _activationFunction->getName(),
        _lossFunction->getName());
}

std::vector<double> Perceptron::getWeights() {
    return _weights;
}

double Perceptron::getBias() {
    return _bias;
}

double Perceptron::_predictSingle(const std::vector<double>& X) {
    double linearOutput = std::inner_product(X.begin(), X.end(), _weights.begin(), 0.0) + _bias;
    return _activationFunction->activate(linearOutput);
}

ActivationFunction<double>* Perceptron::_getActivationFunction(const std::string& activationFunction) {
    if (activationFunction == "sigmoid")
        return new Sigmoid<double>();
    else if (activationFunction == "softmax")
        return new SoftMax<double>();
    else if (activationFunction == "relu")
        return new ReLU<double>();
    else if (activationFunction == "leaky_relu")
        return new LeakyReLU<double>();
    else if (activationFunction == "tanh")
        return new Tanh<double>();

    return nullptr;
}

LossFunction<double>* Perceptron::_getLossFunction(const std::string& lossFunction) {
    if (lossFunction == "squared_loss")
        return new SquaredLoss<double>();

    return nullptr;
}
