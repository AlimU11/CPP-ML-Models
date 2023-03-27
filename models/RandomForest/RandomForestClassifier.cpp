#include "RandomForestClassifier.h"
#include "../DecisionTree/DecisionTreeClassifier.h"
#include "../utils/utils.h"

DecisionTree* RandomForestClassifier::_createTree(int minSamplesSplit, int maxDepth, double minImpurity) {
    return new DecisionTreeClassifier(
        minSamplesSplit,
        maxDepth,
        minImpurity
    );
}

double RandomForestClassifier::_accumulate(const std::vector<double>& y) {
    return utils::mostCommonClass<double>(y);
}
