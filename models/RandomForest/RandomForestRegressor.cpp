#include "RandomForestRegressor.h"
#include "../DecisionTree/DecisionTreeRegressor.h"
#include "../utils/utils.h"

DecisionTree* RandomForestRegressor::_createTree(int minSamplesSplit, int maxDepth, double minImpurity) {
    return new DecisionTreeRegressor(
        minSamplesSplit,
        maxDepth,
        minImpurity
    );
}

double RandomForestRegressor::_accumulate(const std::vector<double>& y) {
    return utils::mean<double>(y);
}