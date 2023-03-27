#pragma once
#include <vector>
#include <tuple>

class LinearRegression {
public:
    LinearRegression();
    ~LinearRegression();

    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& Y);
    double predict(const std::vector<std::vector<double>>& X);

    std::vector<double> weights() { return _weights; }
    double bias() { return _bias; }

private:
    double _predictSingle(const std::vector<double>& X);
    void _optimizeUnivariate(const std::vector<std::vector<double>>& X, const std::vector<double>& Y);
    void _optimizeMultivariate(const std::vector<std::vector<double>>& X, const std::vector<double>& Y);

    double _bias;
    std::vector<double> _weights;
};