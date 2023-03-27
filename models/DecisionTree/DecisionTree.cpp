#include <set>
#include <algorithm>
#include <iterator>
#include "DecisionTree.h"

DecisionTree::DecisionTree(
    const int& minSamplesSplit,
    const int& maxDepth,
    const double& minImpurity) {
    _minSamplesSplit = minSamplesSplit;
    _maxDepth = maxDepth;
    _minImpurity = minImpurity;
    _root = nullptr;
}

DecisionTree::~DecisionTree() {
    delete _root;
}

void DecisionTree::fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
    if (_root != nullptr) {
        delete _root;
    }

    _root = _buildTree(X, y, 0);
}

std::vector<double> DecisionTree::predict(const std::vector<std::vector<double>>& X) {
    std::vector<double> y_pred(X.size());

    for (int i = 0; i < X.size(); i++) {
        y_pred[i] = _predictSingle(X[i]);
    }
    return y_pred;
}

double DecisionTree::_predictSingle(const std::vector<double>& x) {
    Node* node = _root;

    while (node->_left != nullptr && node->_right != nullptr) {
        if (x[node->_feature_idx] < node->_threshold) {
            node = node->_left;
        } else {
            node = node->_right;
        }
    }

    return node->_value;
}

Node* DecisionTree::_buildTree(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y,
    int depth = 0
) {
    std::vector<std::vector<double>> X_left;
    std::vector<std::vector<double>> X_right;
    std::vector<double> y_left;
    std::vector<double> y_right;

    int best_feature_idx = -1;
    double threshold = -1.0;
    double max_impurity = -1;

    if (X.size() >= _minSamplesSplit || depth <= _maxDepth) {
        _findBestSplit(X, y, X_left, X_right, y_left, y_right, best_feature_idx, threshold, max_impurity);
    }

    if (max_impurity > _minImpurity) {
        Node* left = _buildTree(X_left, y_left, depth + 1);
        Node* right = _buildTree(X_right, y_right, depth + 1);

        return new Node(left, right, best_feature_idx, threshold, 0.0);
    }

    return new Node(_leafValue(y));
}

void DecisionTree::_findBestSplit(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y,
    std::vector<std::vector<double>>& X_left,
    std::vector<std::vector<double>>& X_right,
    std::vector<double>& y_left,
    std::vector<double>& y_right,
    int& best_feature_idx,
    double& threshold,
    double& max_impurity
) {
    size_t n_samples = X.size();
    size_t n_features = X[0].size();
    size_t n_classes = std::set<double>(y.begin(), y.end()).size();

    for (int feature_idx = 0; feature_idx < n_features; feature_idx++) {
        std::set<double> features_set;
        std::transform(X.begin(), X.end(), std::inserter(features_set, features_set.begin()), [feature_idx](const std::vector<double>& x) { return x[feature_idx]; });
        std::vector<double> feature_values(features_set.begin(), features_set.end());

        for (double& feature_threshold : feature_values) {
            std::vector<std::vector<double>> X_left_feature;
            std::vector<std::vector<double>> X_right_feature;
            std::vector<double> y_left_feature;
            std::vector<double> y_right_feature;

            for (int i = 0; i < n_samples; i++) {
                if (X[i][feature_idx] < feature_threshold) {
                    X_left_feature.push_back(X[i]);
                    y_left_feature.push_back(y[i]);
                } else {
                    X_right_feature.push_back(X[i]);
                    y_right_feature.push_back(y[i]);
                }
            }

            if (y_left_feature.size() > 0 && y_right_feature.size() > 0) {
                double impurity = _impurity(y, y_left_feature, y_right_feature);

                if (impurity > max_impurity) {
                    max_impurity = impurity;
                    X_left = X_left_feature;
                    X_right = X_right_feature;
                    y_left = y_left_feature;
                    y_right = y_right_feature;
                    best_feature_idx = feature_idx;
                    threshold = feature_threshold;
                }
            }
        }
    }
}
