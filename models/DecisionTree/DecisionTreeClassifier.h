#pragma once
#include "DecisionTree.h"

class DecisionTreeClassifier : public DecisionTree {
public:
    using DecisionTree::DecisionTree;
private:
    double _leafValue(const std::vector<double>& y) override;
    double _impurity(
        const std::vector<double>& y,
        const std::vector<double>& y_left,
        const std::vector<double>& y_right) override;

    double _calculateEntropy(const std::vector<double>& y);
};
