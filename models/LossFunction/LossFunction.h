#pragma once
#include <vector>
#include <cmath>

template <typename T>
class LossFunction {
public:
    virtual ~LossFunction() {}

    virtual T loss(const T& y, const T& y_pred) = 0;
    virtual T gradient(const T& y, const T& y_pred) = 0;

    virtual std::vector<T> loss(const std::vector<T>& y, const T& y_pred);
    virtual std::vector<T> gradient(const std::vector<T>& y, const T& y_pred);

    std::string getName() const { return _name; }

protected:
    std::string _name = "Base Loss Function";
};

template <typename T>
inline std::vector<T> LossFunction<T>::loss(const std::vector<T>& y, const T& y_pred) {
    std::vector<T> result(y.size());
    std::transform(y.begin(), y.end(), result.begin(), [this, &y_pred](const T& y) { return loss(y, y_pred); });
    return result;
}

template <typename T>
inline std::vector<T> LossFunction<T>::gradient(const std::vector<T>& y, const T& y_pred) {
    std::vector<T> result(y.size());
    std::transform(y.begin(), y.end(), result.begin(), [this, &y_pred](const T& y) { return gradient(y, y_pred); });
    return result;
}
