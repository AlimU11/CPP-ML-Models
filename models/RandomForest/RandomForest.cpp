#include <numeric>
#include "RandomForest.h"

RandomForest::RandomForest(
    const int& nEstimators,
    const int& maxFeatures,
    const int& minSamplesSplit,
    const int& maxDepth,
    const double& minImpurity,
    const double& minGain
) {
    _nEstimators = nEstimators;
    _maxFeatures = maxFeatures;
    _minSamplesSplit = minSamplesSplit;
    _maxDepth = maxDepth;
    _minImpurity = minImpurity;
    _minGain = minGain;
}

RandomForest::~RandomForest() {
    for (int i = 0; i < _trees.size(); i++) {
        delete _trees[i];
    }
}

void RandomForest::fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
    size_t n_samples = X.size();
    size_t n_features = X[0].size();

    if (_maxFeatures == -1) {
        _maxFeatures = (int) sqrt((double) n_features);
    }

    for (int i = 0; i < _nEstimators; i++) {
        std::vector<std::vector<double>> X_bootstrap;
        std::vector<double> y_bootstrap;
        std::vector<int> indexes(n_samples);
        std::iota(indexes.begin(), indexes.end(), 0);

        for (int j = 0; j < n_samples; j++) {
            int idx = indexes[rand() % indexes.size()];
            indexes.erase(std::remove(indexes.begin(), indexes.end(), idx), indexes.end());

            X_bootstrap.push_back(X[idx]);
            y_bootstrap.push_back(y[idx]);
        }

        DecisionTree* tree = _createTree(_minSamplesSplit, _maxDepth, _minImpurity);
        tree->fit(X_bootstrap, y_bootstrap);
        _trees.push_back(tree);
    }
}

std::vector<double> RandomForest::predict(const std::vector<std::vector<double>>& X) {
    size_t n_samples = X.size();
    std::vector<double> y_pred(n_samples);

    for (int i = 0; i < n_samples; i++) {
        y_pred[i] = _predictSingle(X[i]);
    }

    return y_pred;
}

double RandomForest::_predictSingle(const std::vector<double>& x) {
    std::vector<double> y_pred(_trees.size());

    for (int i = 0; i < _trees.size(); i++) {
        y_pred[i] = _trees[i]->_predictSingle(x);
    }

    return _accumulate(y_pred);
}