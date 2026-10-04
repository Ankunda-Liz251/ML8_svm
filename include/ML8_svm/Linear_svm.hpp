
//  linear_svm.hpp     
//  LinearSVM is the "main door" of the library. A user only needs this class
//  to train a model and make predictions:
//
//      LinearSVM model(lambda, config);   // 1. choose the settings
//      model.fit(X_train, y_train);       // 2. learn from the data
//      int label = model.predict(x);      // 3. use it
//
//  Behind the scenes, fit() asks the optimizer (M7) to find the best line,
//  and predict() uses the decision function (decision.hpp) to pick a side.
//
//  LABELS must be -1 or +1 (use preprocessing.hpp's encode_binary_labels first).
//  The model is LINEAR: it can only separate data with a straight line/plane.

#pragma once

#include <ML8_svm/hyperplane.hpp>
#include <ML8_svm/optimizer.hpp>   // OptimizerConfig, TrainingHistory
#include <ML8_svm/types.hpp>

#include <cstddef>
#include <vector>

namespace ml8_svm {

class LinearSVM {
public:
    // lambda : regularisation strength (>= 0). Bigger lambda = simpler model
    //          with a wider margin; smaller lambda = fits the training data harder.
    // cfg    : training settings (learning rate, epochs, ...). See optimizer.hpp.
    // Throws InvalidArgument if lambda is negative / not finite,
    //        or learning_rate <= 0, or epochs == 0.
    LinearSVM(double lambda, OptimizerConfig cfg);

    // Learn the hyperplane from training data.
    //   X : one row per sample (all rows must have the same length)
    //   y : one label per row, each -1 or +1, and BOTH classes must appear
    // Throws InvalidArgument / DimensionMismatch for bad input.
    // If fit() throws, the model is left exactly as it was before the call.
    // Calling fit() again replaces the old model (it re-trains from zero).
    void fit(const Matrix& X, const std::vector<int>& y);

    // Score of one sample / many samples (w . x + b). See decision.hpp.
    // Throws NotFitted if fit() has not succeeded yet.
    double decision_function(const Vector& x) const;
    Vector decision_function(const Matrix& X) const;

    // Predicted label (+1 or -1) for one sample / many samples.
    // Throws NotFitted before fit(); DimensionMismatch for wrong feature count.
    int predict(const Vector& x) const;
    std::vector<int> predict(const Matrix& X) const;

    // Read-only access to what the model learned. Throw NotFitted before fit().
    const Hyperplane& hyperplane() const;
    const TrainingHistory& history() const;

    // Row numbers (in the training data) of the support vectors, i.e. the
    // training points on or inside the margin. Throws NotFitted before fit().
    std::vector<size_t> support_vector_indices() const;

private:
    // Stops with NotFitted if the model has not been trained yet.
    void require_fitted(const char* caller) const;

    double lambda_;                 // regularisation strength
    OptimizerConfig config_;        // training settings
    Hyperplane hyperplane_;         // the learned line (w, b)
    TrainingHistory history_{};     // loss values recorded while training
    bool fitted_ = false;           // true once fit() has succeeded

    // A copy of the training data, kept ONLY so that
    // support_vector_indices() can look at the training points again.
    Matrix train_X_;
    std::vector<int> train_y_;
};

}  // namespace ml8_svm