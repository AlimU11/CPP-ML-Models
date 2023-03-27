#pragma once
#include <vector>
#include <unordered_map>

class NaiveBayes {
public:
    NaiveBayes();
    ~NaiveBayes();

    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& Y);
    std::vector<double> predict(const std::vector<std::vector<double>>& X);

private:
    double _predictSingle(const std::vector<double>& X);
    double _calculatePrior(const int& class_idx);
    double _calculateLikelihood(const std::vector<double>& X, const int& class_idx, const int& feature_idx);
    double _calculateMean(const std::vector<double>& X);
    double _calculateVariance(const std::vector<double>& X, const double& mean);

    std::vector<double> _classes;
    std::unordered_map<double, int> _classesFreq;
    std::vector<std::vector<double>> _means;
    std::vector<std::vector<double>> _variances;
    std::vector<double> _priors;

    size_t _num_features;

    const double _epsilon = 1e-4;
};