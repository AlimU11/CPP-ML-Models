#pragma once
#include <vector>
#include <numeric>
#include <map>

namespace utils {
    template <typename T>
    T mostCommonClass(const std::vector<T>& y) {
        std::map<double, int> counts;
        for (int i = 0; i < y.size(); i++) {
            counts[y[i]]++;
        }

        int max_count = 0;
        double max_class = 0;
        for (auto it = counts.begin(); it != counts.end(); it++) {
            if (it->second > max_count) {
                max_count = it->second;
                max_class = it->first;
            }
        }

        return max_class;
    }

    template <typename T>
    T mean(const std::vector<T>& y) {
        return std::accumulate(y.begin(), y.end(), 0.0) / y.size();
    }
}
