#pragma once
#include "ActivationFunction.h"

template <typename T>
class Tanh : public ActivationFunction<T> {
public:
    Tanh() { this->_name = "Tanh"; }

    T activate(const T& X) override;
    T derivate(const T& X) override;
};

template<typename T>
inline T Tanh<T>::activate(const T& X) {
    return 2.0 / (1.0 + std::exp(-2.0 * X)) - 1.0;
}

template<typename T>
inline T Tanh<T>::derivate(const T& X) {
    return 1.0 - std::pow(activate(X), 2);
}
