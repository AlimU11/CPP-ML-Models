#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "LinearRegression/LinearRegression.h"
#include "LogisticRegression/LogisticRegression.h"
#include "KNN/KNNClassifier.h"
#include "KNN/KNNRegressor.h"
#include "NaiveBayes/NaiveBayes.h"
#include "Perceptron/Perceptron.h"
#include "DecisionTree/DecisionTreeClassifier.h"
#include "DecisionTree/DecisionTreeRegressor.h"
#include "RandomForest/RandomForestClassifier.h"
#include "RandomForest/RandomForestRegressor.h"

namespace py = pybind11;

PYBIND11_MODULE(models, m) {
    py::class_<LinearRegression>(m, "LinearRegression")
        .def(py::init<>())
        .def("fit", &LinearRegression::fit)
        .def("predict", &LinearRegression::predict)
        .def_property_readonly("weights", &LinearRegression::weights)
        .def_property_readonly("bias", &LinearRegression::bias);

    py::class_<LogisticRegression>(m, "LogisticRegression")
        .def(py::init<float, int, std::string>(),
            py::arg("learning_rate") = LEARNING_RATE,
            py::arg("n_iter") = MAX_ITER,
            py::arg("solver") = LR_SOLVER)
        .def("fit", &LogisticRegression::fit)
        .def("predict", &LogisticRegression::predict)
        .def("predict_proba", &LogisticRegression::predictProba)
        .def_property_readonly("weights", &LogisticRegression::weights)
        .def_property_readonly("bias", &LogisticRegression::bias)
        .def_property_readonly("learning_rate", &LogisticRegression::learningRate)
        .def_property_readonly("n_iter", &LogisticRegression::maxIter)
        .def_property_readonly("solver", &LogisticRegression::solver)
        .def("get_params", &LogisticRegression::getParams);

    py::class_<KNN>(m, "KNN");

    py::class_<KNNClassifier, KNN>(m, "KNNClassifier")
        .def(py::init<int>(), py::arg("k") = N_NEIGHBORS)
        .def("fit", &KNNClassifier::fit)
        .def("predict", &KNNClassifier::predict)
        .def_property_readonly("k", &KNNClassifier::k);

    py::class_<KNNRegressor, KNN>(m, "KNNRegressor")
        .def(py::init<int>(), py::arg("k") = N_NEIGHBORS)
        .def("fit", &KNNRegressor::fit)
        .def("predict", &KNNRegressor::predict)
        .def_property_readonly("k", &KNNRegressor::k);

    py::class_<NaiveBayes>(m, "NaiveBayes")
        .def(py::init<>())
        .def("fit", &NaiveBayes::fit)
        .def("predict", &NaiveBayes::predict);
    // .def("classes", &NaiveBayes::classes)
    // .def("means", &NaiveBayes::means)
    // .def("variances", &NaiveBayes::variances)
    // .def("priors", &NaiveBayes::priors);

    py::class_<Perceptron>(m, "Perceptron")
        .def(py::init<int, double, std::string, std::string>(),
            py::arg("n_iter") = N_ITERATIONS,
            py::arg("learning_rate") = LEARNING_RATE,
            py::arg("activation_function") = std::string("sigmoid"),
            py::arg("loss_function") = std::string("squared_loss"))
        .def("fit", &Perceptron::fit)
        .def("predict", &Perceptron::predict)
        .def("predict_proba", &Perceptron::predict_proba)
        .def_property_readonly("weights", &Perceptron::getWeights)
        .def_property_readonly("bias", &Perceptron::getBias)
        .def_property_readonly("get_params", &Perceptron::getParameters);

    py::class_<DecisionTreeClassifier>(m, "DecisionTreeClassifier")
        .def(py::init<int, int, double>(),
            py::arg("max_depth") = MAX_DEPTH,
            py::arg("min_samples_split") = MIN_SAMPLES_SPLIT,
            py::arg("min_impurity_decrease") = MIN_IMPURITY)
        .def("fit", &DecisionTreeClassifier::fit)
        .def("predict", &DecisionTreeClassifier::predict);

    py::class_<DecisionTreeRegressor>(m, "DecisionTreeRegressor")
        .def(py::init<int, int, double>(),
            py::arg("max_depth") = MAX_DEPTH,
            py::arg("min_samples_split") = MIN_SAMPLES_SPLIT,
            py::arg("min_impurity_decrease") = MIN_IMPURITY)
        .def("fit", &DecisionTreeRegressor::fit)
        .def("predict", &DecisionTreeRegressor::predict);

    py::class_<RandomForestClassifier>(m, "RandomForestClassifier")
        .def(py::init<int, int, int, int, double, double>(),
            py::arg("n_estimators") = N_ESTIMATORS,
            py::arg("max_features") = MAX_FEATURES,
            py::arg("min_samples_split") = MIN_SAMPLES_SPLIT,
            py::arg("max_depth") = MAX_DEPTH,
            py::arg("min_impurity_decrease") = MIN_IMPURITY,
            py::arg("min_gain") = MIN_GAIN)
        .def("fit", &RandomForestClassifier::fit)
        .def("predict", &RandomForestClassifier::predict);

    py::class_<RandomForestRegressor>(m, "RandomForestRegressor")
        .def(py::init<int, int, int, int, double, double>(),
            py::arg("n_estimators") = N_ESTIMATORS,
            py::arg("max_features") = MAX_FEATURES,
            py::arg("min_samples_split") = MIN_SAMPLES_SPLIT,
            py::arg("max_depth") = MAX_DEPTH,
            py::arg("min_impurity_decrease") = MIN_IMPURITY,
            py::arg("min_gain") = MIN_GAIN)
        .def("fit", &RandomForestRegressor::fit)
        .def("predict", &RandomForestRegressor::predict);
}
