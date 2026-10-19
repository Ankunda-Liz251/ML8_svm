

//  linear_svm.cpp        
//  Implementation of the LinearSVM class.

#include <ML8_svm/linear_svm.hpp>

#include <ML8_svm/decision.hpp>
#include <ML8_svm/errors.hpp>
#include <ML8_svm/support_vectors.hpp>

#include <cmath>     // std::isfinite
#include <string>

namespace ml8_svm {


// Small helper that only this file uses (anonymous namespace = private).
// It checks that the training data is usable BEFORE we start training, so the
// user gets a clear message instead of a crash or nonsense results.

namespace {

void check_training_data(const Matrix& X, const std::vector<int>& y) {
    // 1. There must be at least one sample.
    if (X.empty()) {
        throw InvalidArgument("fit: no training samples (X is empty)");
    }
    // 2. One label per sample.
    if (X.size() != y.size()) {
        throw DimensionMismatch("fit: X has " + std::to_string(X.size()) +
                                " rows but y has " + std::to_string(y.size()) +
                                " labels");
    }
    // 3. Every row must have the same, non-zero number of features,
    //    and no value may be NaN or infinity.
    const size_t n_features = X[0].size();
    if (n_features == 0) {
        throw InvalidArgument("fit: samples have zero features");
    }
    for (size_t i = 0; i < X.size(); ++i) {
        if (X[i].size() != n_features) {
            throw DimensionMismatch("fit: row " + std::to_string(i) +
                                    " has a different number of features");
        }
        for (double value : X[i]) {
            if (!std::isfinite(value)) {
                throw InvalidArgument("fit: row " + std::to_string(i) +
                                      " contains NaN or infinity");
            }
        }
    }
    // 4. Labels must be -1 or +1, and both classes must be present
    //    (an SVM needs two sides to draw a line between).
    bool seen_positive = false;
    bool seen_negative = false;
    for (int label : y) {
        if (label == +1)      seen_positive = true;
        else if (label == -1) seen_negative = true;
        else throw InvalidArgument("fit: labels must be -1 or +1, found " +
                                   std::to_string(label));
    }
    if (!seen_positive || !seen_negative) {
        throw InvalidArgument("fit: training data must contain both classes (-1 and +1)");
    }
}

}  // namespace

// constructor 
LinearSVM::LinearSVM(double lambda, OptimizerConfig cfg)
    : lambda_(lambda), config_(cfg) {
    // Check the settings straight away so mistakes show up early.
    if (!std::isfinite(lambda) || lambda < 0.0) {
        throw InvalidArgument("LinearSVM: lambda must be a finite number >= 0");
    }
    if (!std::isfinite(cfg.learning_rate) || cfg.learning_rate <= 0.0) {
        throw InvalidArgument("LinearSVM: learning_rate must be > 0");
    }
    if (cfg.epochs == 0) {
        throw InvalidArgument("LinearSVM: epochs must be at least 1");
    }
    // tol is "stop early when the loss changes by less than this", so it
    // cannot be negative or NaN. (0 is allowed: it means "never stop early".)
    if (!std::isfinite(cfg.tol) || cfg.tol < 0.0) {
        throw InvalidArgument("LinearSVM: tol must be a finite number >= 0");
    }
}

//  fit 
void LinearSVM::fit(const Matrix& X, const std::vector<int>& y) {
    // Step 1: make sure the data is valid (throws if not).
    check_training_data(X, y);

    // Step 2: start from a flat line: all weights 0 and bias 0.
    // We work on a LOCAL copy, not on the member variables. This way, if the
    // optimizer throws half-way, the old model is not left half-trained.
    Hyperplane trained;
    trained.w = Vector(X[0].size(), 0.0);
    trained.b = 0.0;

    // Step 3: let the optimizer improve the line (gradient descent on the
    // hinge loss + L2 penalty). It changes `trained` and returns the loss log.
    TrainingHistory log = optimize(trained, X, y, lambda_, config_);

    // Step 3b: a learning rate that is too big (or unscaled features) can make
    // the numbers explode into infinity or NaN. A model like that would give
    // nonsense, so we refuse to keep it. The old model stays untouched.
    bool numbers_ok = std::isfinite(trained.b);
    for (double weight : trained.w) {
        if (!std::isfinite(weight)) numbers_ok = false;
    }
    if (!numbers_ok) {
        throw InvalidArgument(
            "fit: training diverged (weights became NaN or infinity). "
            "Try a smaller learning_rate or scale the features first.");
    }

    // Step 4: everything worked, so now we store the results.
    hyperplane_ = trained;
    history_    = log;
    train_X_    = X;       // keep a copy for support_vector_indices()
    train_y_    = y;
    fitted_     = true;
}

//  scores
double LinearSVM::decision_function(const Vector& x) const {
    require_fitted("decision_function");
    return ml8_svm::decision_function(hyperplane_, x);   // free function in decision.hpp
}

Vector LinearSVM::decision_function(const Matrix& X) const {
    require_fitted("decision_function");
    return ml8_svm::decision_function(hyperplane_, X);
}

//  predictions 
int LinearSVM::predict(const Vector& x) const {
    require_fitted("predict");
    // prediction = the sign of the score
    return to_label(ml8_svm::decision_function(hyperplane_, x));
}

std::vector<int> LinearSVM::predict(const Matrix& X) const {
    require_fitted("predict");
    // Compute all scores first (this also checks every row's size) ...
    const Vector scores = ml8_svm::decision_function(hyperplane_, X);
    // ... then convert each score into a label.
    std::vector<int> labels;
    labels.reserve(scores.size());
    for (double s : scores) labels.push_back(to_label(s));
    return labels;
}

// accessors 
const Hyperplane& LinearSVM::hyperplane() const {
    require_fitted("hyperplane");
    return hyperplane_;
}

const TrainingHistory& LinearSVM::history() const {
    require_fitted("history");
    return history_;
}

std::vector<size_t> LinearSVM::support_vector_indices() const {
    require_fitted("support_vector_indices");
    // M8's function does the real work; we just give it the training data.
    return find_support_vectors(hyperplane_, train_X_, train_y_);
}

//  private helper
void LinearSVM::require_fitted(const char* caller) const {
    if (!fitted_) {
        throw NotFitted(std::string("LinearSVM::") + caller +
                        ": the model is not trained yet - call fit() first");
    }
}

}  // namespace ml8_svm