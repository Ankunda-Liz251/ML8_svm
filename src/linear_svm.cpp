
//  linear_svm.cpp  
//  The class compiles and links. Nothing can be trained yet, so every method
//  that needs a trained model throws NotFitted, as the plan requires.

#include <ML8_svm/linear_svm.hpp>
#include <ML8_svm/errors.hpp>

namespace ml8_svm {

// Just store the settings for now (validation comes in Week 2).
LinearSVM::LinearSVM(double lambda, OptimizerConfig cfg)
    : lambda_(lambda), config_(cfg) {}

void LinearSVM::fit(const Matrix&, const std::vector<int>&) {
    throw SvmError("LinearSVM::fit: not implemented yet (Week 2)");
}

// Because fitted_ is always false in Week 1, these all throw NotFitted.
double LinearSVM::decision_function(const Vector&) const { require_fitted("decision_function"); return 0.0; }
Vector LinearSVM::decision_function(const Matrix&) const { require_fitted("decision_function"); return {}; }
int LinearSVM::predict(const Vector&) const { require_fitted("predict"); return 0; }
std::vector<int> LinearSVM::predict(const Matrix&) const { require_fitted("predict"); return {}; }
const Hyperplane& LinearSVM::hyperplane() const { require_fitted("hyperplane"); return hyperplane_; }
const TrainingHistory& LinearSVM::history() const { require_fitted("history"); return history_; }
std::vector<size_t> LinearSVM::support_vector_indices() const { require_fitted("support_vector_indices"); return {}; }

void LinearSVM::require_fitted(const char* caller) const {
    if (!fitted_) {
        throw NotFitted(std::string("LinearSVM::") + caller +
                        ": the model is not trained yet - call fit() first");
    }
}

}  // namespace ml8_svm