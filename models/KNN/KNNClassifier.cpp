#pragma once

#include <map>
#include "KNNClassifier.h"

double KNNClassifier::_accumulate(const std::vector<double>& Y) {
    return _majorityVote(Y);
}

double KNNClassifier::_majorityVote(const std::vector<double>& Y) {
    std::vector<double> uniqueY = std::vector<double>(Y.cbegin(), Y.cend());
    std::map<double, int> counts;

    for (double y : uniqueY) {
        counts[y] = 0;
    }

    for (auto& y : Y) {
        counts[y]++;
    }

    int maxCount = 0;
    double maxIndex = 0.0;

    for (auto& count : counts) {
        if (count.second > maxCount) {
            maxIndex = count.first;
            maxCount = count.second;
        }
    }

    return maxIndex;
}