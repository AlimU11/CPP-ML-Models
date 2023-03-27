#pragma once
#include "ActivationFunction.h"

template <typename T>
class ReLU : public ActivationFunction<T> {
public:
    ReLU() { this->_name = "ReLU"; }

    T activate(const T& X) override;
    T derivate(const T& X) override;
};

template<typename T>
inline T ReLU<T>::activate(const T& X) {
    return X > 0 ? X : 0;
}

template<typename T>
inline T ReLU<T>::derivate(const T& X) {
    return X > 0 ? 1 : 0;
}
