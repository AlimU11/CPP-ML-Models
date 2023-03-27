#pragma once
#include <vector>
#include <cmath>
#include <algorithm>

template <typename T>
class ActivationFunction {
public:
    virtual ~ActivationFunction() {}

    virtual T activate(const T& X) = 0;
    virtual T derivate(const T& X) = 0;

    virtual std::vector<T> activate(const std::vector<T>& X);
    virtual std::vector<T> derivate(const std::vector<T>& X);

    inline std::string getName() const { return _name; }

protected:
    std::string _name = "Base Activation Function";
};

template <typename T>
inline std::vector<T> ActivationFunction<T>::activate(const std::vector<T>& X) {
    std::vector<T> result(X.size());
    std::transform(X.begin(), X.end(), result.begin(), [this](const T& x) { return activate(x); });
    return result;
}

template <typename T>
inline std::vector<T> ActivationFunction<T>::derivate(const std::vector<T>& X) {
    std::vector<T> result(X.size());
    std::transform(X.begin(), X.end(), result.begin(), [this](const T& x) { return derivate(x); });
    return result;
}
