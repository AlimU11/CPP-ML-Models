#pragma once
#include <vector>
#include <string>
#include <tuple>
#include "LRSolver.h"

#define LEARNING_RATE 0.01
#define MAX_ITER 1000
#define LR_SOLVER "sgd"

class LogisticRegression {
public:
    LogisticRegression(
        const double& learning_rate = LEARNING_RATE,
        const int& max_iter = MAX_ITER,
        const std::string& solver = LR_SOLVER);
    ~LogisticRegression();

    void fit(const std::vector<std::vector<double>>& X, const std::vector<int>& Y);
    std::vector<int> predict(const std::vector<std::vector<double>>& X);
    std::vector<double> predictProba(const std::vector<std::vector<double>>& X);

    std::vector<double> weights() { return _weights; }
    double bias() { return _bias; }
    double learningRate() { return _learningRate; }
    int maxIter() { return _maxIter; }
    std::string solver() {
        return _solver == LRSolver::SGD ? "sgd" : _solver == LRSolver::SAG ? "sag"
            : "saga";
    }

    std::tuple<double, int, std::string> getParams();

private:
    void _sgd(
        const std::vector<std::vector<double>>& X,
        const std::vector<int>& Y);
    void _sag(
        const size_t& n_samples,
        const size_t& n_features,
        std::vector<double>& sum_grad);
    void _saga(
        const size_t& n_samples,
        const size_t& n_features,
        std::vector<double>& sum_grad,
        std::vector<double>& prev_sum_grad);
    void _sagHelper(
        const std::vector<std::vector<double>>& X,
        const std::vector<int>& Y);

    double _predictSingle(const std::vector<double>& X);
    double _sigmoid(double& y_pred);
    double _avgGradDiff(const std::vector<std::vector<double>>& grad);
    LRSolver _getSolver(const std::string& solver);

    double _bias;
    std::vector<double> _weights;

    LRSolver _solver;
    double _learningRate;
    int _maxIter;
};