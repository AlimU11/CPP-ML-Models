#pragma once
#include "RandomForest.h"

class RandomForestRegressor : public RandomForest {
public:
    using RandomForest::RandomForest;
private:
    DecisionTree* _createTree(int minSamplesSplit, int maxDepth, double minImpurity) override;
    double _accumulate(const std::vector<double>& y) override;
};
