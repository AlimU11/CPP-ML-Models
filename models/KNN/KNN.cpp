#pragma once
#include "KNN.h"
#include <cmath>
#include <numeric>
#include <algorithm>

KNN::KNN(int k) {
    _k = k;
}

KNN::~KNN() {}

void KNN::fit(const std::vector<std::vector<double>>& X, const std::vector<double>& Y) {
    _X = X;
    _Y = Y;
}

std::vector<double> KNN::predict(const std::vector<std::vector<double>>& X) {
    std::vector<double> predictions = std::vector<double>(X.size(), 0.0);

    for (size_t i = 0; i < X.size(); i++) {
        predictions[i] = _predictSingle(X[i]);
    }

    return predictions;
}

double KNN::_predictSingle(const std::vector<double>& X) {
    std::vector<double> distances = std::vector<double>(_X.size(), 0.0);

    for (size_t i = 0; i < _X.size(); i++) {
        distances[i] = _calculateDistance(X, _X[i]);
    }

    std::vector<int> indices = std::vector<int>(distances.size(), -1);

    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&distances](int i1, int i2) { return distances[i1] < distances[i2]; });

    std::vector<double> Y = std::vector<double>(_k, 0.0);

    for (size_t i = 0; i < _k; i++) {
        Y[i] = _Y[indices[i]];
    }

    return _accumulate(Y);
}

double KNN::_calculateDistance(const std::vector<double>& X1, const std::vector<double>& X2) {
    double distance = 0.0;

    for (size_t i = 0; i < X1.size(); i++) {
        distance += std::pow(X1[i] - X2[i], 2);
    }

    return std::sqrt(distance);
}
