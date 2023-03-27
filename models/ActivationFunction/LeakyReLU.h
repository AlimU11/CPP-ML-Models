#pragma once
#include "ActivationFunction.h"

template <typename T>
class LeakyReLU : public ActivationFunction<T> {
public:
    LeakyReLU(double alpha = 0.3) : _alpha(alpha) { this->_name = "LeakyReLU"; }

    T activate(const T& X) override;
    T derivate(const T& X) override;

private:
    double _alpha;
};

template<typename T>
inline T LeakyReLU<T>::activate(const T& X) {
    return X > 0 ? X : _alpha * X;
}

template<typename T>
inline T LeakyReLU<T>::derivate(const T& X) {
    return X > 0 ? 1 : _alpha;
}
