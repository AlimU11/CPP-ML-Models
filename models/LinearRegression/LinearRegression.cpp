#pragma once
#include "LinearRegression.h"
#include <cmath>

#define N_ITER 1000
#define LEARNING_RATE 0.01

LinearRegression::LinearRegression() {
    _bias = 0;
}

LinearRegression::~LinearRegression() {}

void LinearRegression::fit(const std::vector<std::vector<double>>& X, const std::vector<double>& Y) {
    if (X[0].size() == 1) {
        _optimizeUnivariate(X, Y);
    } else {
        _optimizeMultivariate(X, Y);
    }
}

double LinearRegression::predict(const std::vector<std::vector<double>>& X) {
    return 0.0;
}

double LinearRegression::_predictSingle(const std::vector<double>& X) {
    return 0.0;
}

void LinearRegression::_optimizeUnivariate(const std::vector<std::vector<double>>& X, const std::vector<double>& Y) {
    const size_t n_samples = X.size();

    _weights = std::vector<double>(1, 0.0);

    double X_sum = 0.0;
    double X_mean = 0.0;
    double X_squared_diff = 0.0;
    double X_std = 0.0;

    double Y_sum = 0.0;
    double Y_mean = 0.0;
    double Y_squared_diff = 0.0;
    double Y_std = 0.0;

    double cov = 0.0;
    double corr = 0.0;

    for (size_t i = 0; i < n_samples; i++) {
        X_sum += X[i][0];
        Y_sum += Y[i];
    }

    X_mean = X_sum / n_samples;
    Y_mean = Y_sum / n_samples;

    for (size_t i = 0; i < n_samples; i++) {
        double X_diff = X[i][0] - X_mean;
        double Y_diff = Y[i] - Y_mean;

        X_squared_diff += std::pow(X_diff, 2);
        Y_squared_diff += std::pow(Y_diff, 2);

        cov += X_diff * Y_diff;
    }

    X_std = std::sqrt(X_squared_diff / (n_samples - 1));
    Y_std = std::sqrt(Y_squared_diff / (n_samples - 1));

    cov /= (n_samples - 1);
    corr = cov / (X_std * Y_std);

    _weights[0] = corr * Y_std / X_std;
    _bias = Y_mean - _weights[0] * X_mean;
}

void LinearRegression::_optimizeMultivariate(const std::vector<std::vector<double>>& X, const std::vector<double>& Y) {
    const size_t n_samples = X.size();
    const size_t n_features = X[0].size();

    _weights = std::vector<double>(n_features, 0.0);

    for (int epoch = 0; epoch < N_ITER; epoch++) {
        std::vector<double> y_pred(n_samples, 0.0);

        for (size_t i = 0; i < n_samples; i++) {
            for (size_t j = 0; j < n_features; j++) {
                y_pred[i] += _weights[j] * X[i][j];
            }
            y_pred[i] += _bias;
        }

        std::vector<double> weights_gradient(n_features, 0.0);
        double bias_gradient = 0.0;

        for (size_t i = 0; i < n_samples; i++) {
            double diff = y_pred[i] - Y[i];

            for (size_t j = 0; j < n_features; j++) {
                weights_gradient[j] += diff * X[i][j];
            }

            bias_gradient += diff;
        }

        for (size_t j = 0; j < n_features; j++) {
            _weights[j] -= LEARNING_RATE * weights_gradient[j] / n_samples;
        }

        _bias -= LEARNING_RATE * bias_gradient / n_samples;
    }
}
