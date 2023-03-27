#pragma once
#define _USE_MATH_DEFINES
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <set>
#include <iterator>
#include "NaiveBayes.h"

NaiveBayes::NaiveBayes() {
    _num_features = -1;
}

NaiveBayes::~NaiveBayes() {}

void NaiveBayes::fit(const std::vector<std::vector<double>>& X, const std::vector<double>& Y) {
    const size_t n_samples = X.size();
    _num_features = X[0].size();

    std::set<double> classes_set(Y.begin(), Y.end());

    _classes = std::vector<double>(classes_set.begin(), classes_set.end());
    _means = std::vector<std::vector<double>>(_classes.size(), std::vector<double>(_num_features, 0.0));
    _variances = std::vector<std::vector<double>>(_classes.size(), std::vector<double>(_num_features, 0.0));
    _priors = std::vector<double>(_classes.size(), 0.0);

    for (double y : Y) {
        _classesFreq[y]++;
    }

    for (int class_idx = 0; class_idx < _classes.size(); class_idx++) {
        std::vector<std::vector<double>> X_class;

        for (int j = 0; j < n_samples; j++) {
            if (Y[j] == _classes[class_idx]) {
                X_class.push_back(X[j]);
            }
        }

        for (int feature_idx = 0; feature_idx < _num_features; feature_idx++) {
            std::vector<double> feature_col;
            for (int row = 0; row < X_class.size(); row++) {
                feature_col.push_back(X_class[row][feature_idx]);
            }

            _means[class_idx][feature_idx] = _calculateMean(feature_col);
            _variances[class_idx][feature_idx] = _calculateVariance(feature_col, _means[class_idx][feature_idx]);
        }

        _priors[class_idx] = static_cast<double>(X_class.size()) / n_samples;
    }
}

std::vector<double> NaiveBayes::predict(const std::vector<std::vector<double>>& X) {
    std::vector<double> y_pred;
    y_pred.reserve(X.size());

    for (int i = 0; i < X.size(); i++) {
        y_pred.push_back(_predictSingle(X[i]));
    }

    return y_pred;
}

double NaiveBayes::_predictSingle(const std::vector<double>& X) {
    std::vector<double> posteriors;

    for (int class_idx = 0; class_idx < _classes.size(); class_idx++) {
        double posterior = _calculatePrior(class_idx);

        for (int feature_idx = 0; feature_idx < _num_features; feature_idx++) {
            posterior *= _calculateLikelihood(X, class_idx, feature_idx);
        }

        posteriors.push_back(posterior);
    }

    return _classes[std::distance(posteriors.begin(), std::max_element(posteriors.begin(), posteriors.end()))];
}

double NaiveBayes::_calculatePrior(const int& classIndex) {
    return _priors[classIndex];
}

double NaiveBayes::_calculateLikelihood(const std::vector<double>& X, const int& classIndex, const int& featureIndex) {
    double mean = _means[classIndex][featureIndex];
    double variance = _variances[classIndex][featureIndex];
    double coefficient = 1 / sqrt(2 * M_PI * variance + _epsilon);
    double exponent = exp(-pow(X[featureIndex] - mean, 2) / (2 * variance + _epsilon));

    return coefficient * exponent;
}

double NaiveBayes::_calculateMean(const std::vector<double>& X) {
    return std::accumulate(X.begin(), X.end(), 0.0) / X.size();
}

double NaiveBayes::_calculateVariance(const std::vector<double>& X, const double& mean) {
    double sum = 0.0;
    for (int i = 0; i < X.size(); i++) {
        sum += pow(X[i] - mean, 2);
    }

    return sum / X.size();
}
