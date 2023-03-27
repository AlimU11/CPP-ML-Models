#pragma once
#include "LossFunction.h"

template <typename T>
class SquaredLoss : public LossFunction<T> {
public:
    SquaredLoss() { this->_name = "SquaredLoss"; }
    T loss(const T& y, const T& y_pred) override;
    T gradient(const T& y, const T& y_pred) override;
};

template <typename T>
inline T SquaredLoss<T>::loss(const T& y, const T& y_pred) {
    return std::pow(y - y_pred, 2);
}

template <typename T>
inline T SquaredLoss<T>::gradient(const T& y, const T& y_pred) {
    return -(y - y_pred);
}
