#pragma once

#include <cmath>
#include <numeric>
#include <algorithm>
#include "KNNRegressor.h"

double KNNRegressor::_accumulate(const std::vector<double>& Y) {
    return std::accumulate(Y.begin(), Y.end(), 0.0) / Y.size();
}
