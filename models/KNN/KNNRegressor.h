#pragma once

#include "KNN.h"

class KNNRegressor : public KNN {
public:
    using KNN::KNN;

private:
    double _accumulate(const std::vector<double>& Y) override;
};