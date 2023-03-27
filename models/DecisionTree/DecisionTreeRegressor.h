#pragma once
#include "DecisionTree.h"
class DecisionTreeRegressor : public DecisionTree {
public:
    using DecisionTree::DecisionTree;

private:
    double _leafValue(const std::vector<double>& y);
    double _impurity(
        const std::vector<double>& y,
        const std::vector<double>& y_left,
        const std::vector<double>& y_right
    );

    double _variance(const std::vector<double>& y);
};
