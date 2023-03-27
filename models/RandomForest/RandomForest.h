#pragma once
#include <vector>
#include "../DecisionTree/DecisionTree.h"
#define N_ESTIMATORS 25
#define MAX_FEATURES -1
#define MIN_GAIN 0.0

class RandomForest {
public:
    RandomForest(
        const int& nEstimators = N_ESTIMATORS,
        const int& maxFeatures = MAX_FEATURES,
        const int& minSamplesSplit = MIN_SAMPLES_SPLIT,
        const int& maxDepth = MAX_DEPTH,
        const double& minImpurity = MIN_IMPURITY,
        const double& minGain = MIN_GAIN
    );
    ~RandomForest();
    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y);
    std::vector<double> predict(const std::vector<std::vector<double>>& X);

private:
    virtual DecisionTree* _createTree(int minSamplesSplit, int maxDepth, double minImpurity) = 0;
    virtual double _accumulate(const std::vector<double>& y) = 0;
    double _predictSingle(const std::vector<double>& x);

    std::vector<DecisionTree*> _trees;

    int _minSamplesSplit;
    int _maxDepth;
    double _minImpurity;

    int _nEstimators;
    int _maxFeatures;
    double _minGain;
};

