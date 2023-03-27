#pragma once
#include "ActivationFunction.h"
#include <iterator>
#include <numeric>

template <typename T>
class SoftMax : public ActivationFunction<T> {
public:
    SoftMax() { this->_name = "SoftMax"; }

    T activate(const T& X) override;
    T derivate(const T& X) override;

    std::vector<T> activate(const std::vector<T>& X) override;
    std::vector<T> derivate(const std::vector<T>& X) override;
};

template<typename T>
inline T SoftMax<T>::activate(const T& X) {
    return (T) 1.0;
}

template<typename T>
inline T SoftMax<T>::derivate(const T& X) {
    return (T) 0.0;
}

template<typename T>
inline std::vector<T> SoftMax<T>::activate(const std::vector<T>& X) {
    T max = *std::max_element(X.begin(), X.end());

    std::vector<T> exps;
    std::transform(X.begin(), X.end(), std::back_inserter(exps), [max](T x) { return std::exp(x - max); });

    T sum = std::accumulate(exps.begin(), exps.end(), 0.0);

    std::vector<T> result(X.size());
    std::transform(exps.begin(), exps.end(), result.begin(), [sum](T x) { return x / sum; });

    return result;
}

template<typename T>
inline std::vector<T> SoftMax<T>::derivate(const std::vector<T>& X) {
    std::vector<T> p = activate(X);
    std::vector<T> result(X.size());

    std::transform(p.begin(), p.end(), result.begin(), [](T p) { return p * (1 - p); });

    return result;
}
