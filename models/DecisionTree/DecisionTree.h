#pragma once
#include <vector>
#include "Node.h"
#define MIN_SAMPLES_SPLIT 2
#define MAX_DEPTH 100
#define MIN_IMPURITY 1e-7

class DecisionTree {
public:
    DecisionTree(
        const int& minSamplesSplit = MIN_SAMPLES_SPLIT,
        const int& maxDepth = MAX_DEPTH,
        const double& minImpurity = MIN_IMPURITY
    );
    ~DecisionTree();
    virtual void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y);
    std::vector<double> predict(const std::vector<std::vector<double>>& X);

private:
    friend class RandomForest;

    double _predictSingle(const std::vector<double>& x);

    Node* _buildTree(
        const std::vector<std::vector<double>>& X,
        const std::vector<double>& y,
        int depth
    );

    void _findBestSplit(
        const std::vector<std::vector<double>>& X,
        const std::vector<double>& y,
        std::vector<std::vector<double>>& X_left,
        std::vector<std::vector<double>>& X_right,
        std::vector<double>& y_left,
        std::vector<double>& y_right,
        int& best_feature_idx,
        double& threshold,
        double& max_impurity
    );

    virtual double _leafValue(const std::vector<double>& y) = 0;

    virtual double _impurity(
        const std::vector<double>& y,
        const std::vector<double>& y_left,
        const std::vector<double>& y_right
    ) = 0;

    int _minSamplesSplit;
    int _maxDepth;
    double _minImpurity;

    Node* _root;
};
