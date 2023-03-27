#pragma once
#include "DecisionTree.h"
class Node {
public:
    Node(Node* left, Node* right, const int& feature_idx, const double& threshold, const double& value);
    Node(const double& value);
    ~Node();

private:
    Node* _left;
    Node* _right;

    int _feature_idx;
    double _threshold;
    double _value;

    friend class DecisionTree;
};

inline Node::Node(
    Node* left,
    Node* right,
    const int& feature_idx,
    const double& threshold,
    const double& value) {
    _feature_idx = feature_idx;
    _threshold = threshold;
    _left = left;
    _right = right;
    _value = value;
}

inline Node::Node(const double& value) {
    _feature_idx = -1;
    _threshold = -1;
    _left = nullptr;
    _right = nullptr;
    _value = value;
}

inline Node::~Node() {
    delete _left;
    delete _right;
}
