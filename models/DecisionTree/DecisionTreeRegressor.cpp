#include <numeric>
#include "DecisionTreeRegressor.h"

double DecisionTreeRegressor::_leafValue(const std::vector<double>& y) {
    return std::accumulate(y.begin(), y.end(), 0.0) / y.size();
}

double DecisionTreeRegressor::_impurity(
    const std::vector<double>& y,
    const std::vector<double>& y_left,
    const std::vector<double>& y_right) {
    double variance_y = _variance(y);
    double variance_y_left = _variance(y_left);
    double variance_y_right = _variance(y_right);

    double y_left_ratio = (double) y_left.size() / y.size();
    double y_right_ratio = (double) y_right.size() / y.size();

    return variance_y - y_left_ratio * variance_y_left - y_right_ratio * variance_y_right;
}

double DecisionTreeRegressor::_variance(const std::vector<double>& y) {
    double mean = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
    double variance = 0.0;

    for (int i = 0; i < y.size(); i++) {
        variance += (y[i] - mean) * (y[i] - mean);
    }

    return variance / y.size();
}
