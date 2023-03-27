#pragma once
#include "KNN.h"

class KNNClassifier : public KNN {
public:
    using KNN::KNN;

private:
    double _accumulate(const std::vector<double>& Y) override;
    double _majorityVote(const std::vector<double>& Y);
};