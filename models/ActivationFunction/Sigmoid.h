#pragma once
#include "ActivationFunction.h"

template <typename T>
class Sigmoid : public ActivationFunction<T> {
public:
    Sigmoid() { this->_name = "Sigmoid"; }

    T activate(const T& X) override;
    T derivate(const T& X) override;
};

template<typename T>
inline T Sigmoid<T>::activate(const T& X) {
    return (T) 1.0 / ((T) 1.0 + exp(-X));
}

template<typename T>
inline T Sigmoid<T>::derivate(const T& X) {
    return activate(X) * (1 - activate(X));
}
