// metrics.cpp - STUB (Week 1). Real implementation arrives in Week 2.
#include <ML8_svm/metrics.hpp>

#include <stdexcept>

namespace ml8_svm {

namespace {
[[noreturn]] void not_implemented(const char* fn) {
    throw std::logic_error(std::string("metrics: ") + fn +
                           " is not implemented yet (Week 2)");
}
}  // namespace

ConfusionMatrix confusion_matrix(const std::vector<int>&,
                                 const std::vector<int>&) {
    not_implemented("confusion_matrix");
}

double accuracy(const std::vector<int>&, const std::vector<int>&) {
    not_implemented("accuracy");
}

double precision(const std::vector<int>&, const std::vector<int>&) {
    not_implemented("precision");
}

double recall(const std::vector<int>&, const std::vector<int>&) {
    not_implemented("recall");
}

double f1_score(const std::vector<int>&, const std::vector<int>&) {
    not_implemented("f1_score");
}

std::string classification_report(const std::vector<int>&,
                                  const std::vector<int>&) {
    not_implemented("classification_report");
}

}  // namespace ml8_svm