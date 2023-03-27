#include <map>
#include <iostream>
#include "DecisionTreeClassifier.h"
#include "../utils/utils.h"

double DecisionTreeClassifier::_leafValue(const std::vector<double>& y) {
    return utils::mostCommonClass<double>(y);
}

double DecisionTreeClassifier::_impurity(
    const std::vector<double>& y,
    const std::vector<double>& y_left,
    const std::vector<double>& y_right) {
    size_t n = y_left.size() / y.size();

    double entropy = _calculateEntropy(y);
    double entropy_left = _calculateEntropy(y_left);
    double entropy_right = _calculateEntropy(y_right);

    return entropy - n * entropy_left - (1 - n) * entropy_right;
}

double DecisionTreeClassifier::_calculateEntropy(const std::vector<double>& y) {
    std::map<double, int> counts;
    for (int i = 0; i < y.size(); i++) {
        counts[y[i]]++;
    }

    double entropy = 0;
    for (auto it = counts.begin(); it != counts.end(); it++) {
        double p = it->second / (double) y.size();
        entropy -= p * log2(p);
    }

    return entropy;
}
