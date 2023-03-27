#pragma once
#include <vector>
#define N_NEIGHBORS 5

class KNN {
public:
    KNN(int k = N_NEIGHBORS);
    ~KNN();

    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& Y);
    std::vector<double> predict(const std::vector<std::vector<double>>& X);

    int k() const { return _k; }

protected:
    double _predictSingle(const std::vector<double>& X);
    double _calculateDistance(const std::vector<double>& X1, const std::vector<double>& X2);
    virtual double _accumulate(const std::vector<double>& Y) = 0;

    int _k;
    std::vector<std::vector<double>> _X;
    std::vector<double> _Y;
};