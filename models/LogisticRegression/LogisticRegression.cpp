#pragma once
#include "LogisticRegression.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <iostream>

LogisticRegression::LogisticRegression(const double& learning_rate, const int& max_iter, const std::string& solver) {
    _learningRate = learning_rate;
    _maxIter = max_iter;
    _bias = 0.0;
    _getSolver(solver);
}

LogisticRegression::~LogisticRegression() {}

void LogisticRegression::fit(const std::vector<std::vector<double>>& X, const std::vector<int>& Y) {
    _weights = std::vector<double>(X[0].size(), 0.0);

    switch (_solver) {
        case LRSolver::SGD:
            _sgd(X, Y);
            break;
        case LRSolver::SAG:
            _sagHelper(X, Y);
            break;
        case LRSolver::SAGA:
            _sagHelper(X, Y);
            break;
        default:
            throw std::invalid_argument("Unsupported solver");
    }
}

std::vector<int> LogisticRegression::predict(const std::vector<std::vector<double>>& X) {
    std::vector<int> y_pred(X.size());

    for (int i = 0; i < X.size(); i++) {
        y_pred[i] = _predictSingle(X[i]) >= 0.5 ? 1 : 0;
    }

    return y_pred;
}

std::vector<double> LogisticRegression::predictProba(const std::vector<std::vector<double>>& X) {
    std::vector<double> y_pred(X.size());

    for (int i = 0; i < X.size(); i++) {
        y_pred[i] = _predictSingle(X[i]);
    }

    return y_pred;
}

std::tuple<double, int, std::string> LogisticRegression::getParams() {
    return std::tuple<double, int, std::string>(_learningRate, _maxIter, solver());
}

void LogisticRegression::_sgd(const std::vector<std::vector<double>>& X, const std::vector<int>& Y) {
    const size_t n_samples = X.size();

    for (int i = 0; i < _maxIter; i++) {
        for (int row = 0; row < n_samples; row++) {
            const double y_pred = _predictSingle(X[row]);
            const double error = Y[row] - y_pred;

            _bias += _learningRate * error;

            for (int k = 0; k < X[row].size(); k++) {
                _weights[k] += _learningRate * error * X[row][k];
            }
        }
    }
}

void LogisticRegression::_sag(const size_t& n_samples, const size_t& n_features, std::vector<double>& sum_grad) {
    std::transform(_weights.begin(), _weights.end(), sum_grad.begin(), _weights.begin(), [&](double& w, double& sg) { return w + _learningRate * sg / n_samples; });
}

void LogisticRegression::_saga(const size_t& n_samples, const size_t& n_features, std::vector<double>& sum_grad, std::vector<double>& prev_sum_grad) {
    for (int k = 0; k < n_features; k++) {
        _weights[k] += _learningRate * ((sum_grad[k] + prev_sum_grad[k]) / n_samples);
    }
}

void LogisticRegression::_sagHelper(const std::vector<std::vector<double>>& X, const std::vector<int>& Y) {
    const size_t n_samples = X.size();
    const size_t n_features = X[0].size();

    std::vector<std::vector<double>> grad(n_samples, std::vector<double>(n_features, 0.0));
    std::vector<double> sum_grad(n_features, 0.0);
    std::vector<double> prev_sum_grad(n_features, 0.0);

    for (int i = 0; i < _maxIter; i++) {
        for (int j = 0; j < n_samples; j++) {
            const double y_pred = _predictSingle(X[j]);
            const double error = Y[j] - y_pred;

            for (int k = 0; k < n_features; k++) {
                const double curr_grad = error * X[j][k];
                sum_grad[k] -= prev_sum_grad[k];
                sum_grad[k] += curr_grad;
                prev_sum_grad[k] = grad[j][k];
                grad[j][k] = curr_grad;
            }
        }

        switch (_solver) {

            case LRSolver::SAG:
                _sag(n_samples, n_features, sum_grad);
                break;

            case LRSolver::SAGA:
                _saga(n_samples, n_features, sum_grad, prev_sum_grad);
                break;
        }

        _bias += _learningRate * _avgGradDiff(grad);
    }
}

double LogisticRegression::_predictSingle(const std::vector<double>& X) {
    double y_pred = _bias;

    for (int i = 0; i < X.size(); i++) {
        y_pred += X[i] * _weights[i];
    }

    return _sigmoid(y_pred);
}

double LogisticRegression::_sigmoid(double& y_pred) {
    return 1.0 / (1.0 + exp(-y_pred));
}

double LogisticRegression::_avgGradDiff(const std::vector<std::vector<double>>& grad) {
    double sum = 0.0;

    for (int k = 0; k < grad[0].size(); k++) {
        double col_sum = 0.0;

        for (int i = 0; i < grad.size(); i++) {
            col_sum += grad[i][k];
        }

        sum += std::pow(col_sum / grad.size(), 2.0);
    }

    return sum;
}

LRSolver LogisticRegression::_getSolver(const std::string& solver) {
    if (solver == "sgd") {
        return LRSolver::SGD;
    } else if (solver == "sag") {
        return LRSolver::SAG;
    } else if (solver == "saga") {
        return LRSolver::SAGA;
    }

    throw std::invalid_argument("Invalid solver");
}
