

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
//  FULL EXAMPLE
//      Matrix X = {{2, 2}, {3, 3}, {-2, -2}, {-3, -3}};
//      std::vector<int> y = {+1, +1, -1, -1};
//      OptimizerConfig cfg{0.1, 500, 4, 1e-9, false, 42};
//      LinearSVM model(0.1, cfg);
//      model.fit(X, y);
//      int label = model.predict(Vector{1.0, 1.5});     // +1
//
//  RULES TO REMEMBER
//    * LABELS must be -1 or +1 (use preprocessing.hpp's encode_binary_labels
//      first), and BOTH classes must appear in the training data.
//    * Scale the features before training (scaler.hpp). Unscaled data or a
//      learning rate that is too big can make training blow up; fit() then
//      throws instead of keeping a broken model.
//    * The model is LINEAR: it can only separate data with a straight line
//      (or flat plane). Curved boundaries (kernels) are out of scope.
//    * The class can be copied. A copy keeps its own trained line and does not
//      change when the original is trained again.
//
//  COST
//  fit() validates the data in O(n * d), then runs the optimizer; its cost
//  is the optimizer's cost (about epochs * n * d for full-batch training).
//  predict() costs O(d) per sample. The model keeps a copy of the training
//  data (O(n * d) memory) so that support_vector_indices() can use it.
//  Here n = number of samples and d = number of features.

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
    //          0 is allowed (no regularisation).
    // cfg    : training settings (learning rate, epochs, ...). See optimizer.hpp.
    // Throws InvalidArgument if lambda is negative or not finite,
    //        or learning_rate <= 0 (or not finite), or epochs == 0,
    //        or tol is negative or not finite.
    // The model starts "not fitted": nothing can be predicted until fit() works.
    LinearSVM(double lambda, OptimizerConfig cfg);

    // Learn the hyperplane from training data.
    //   X : one row per sample (all rows must have the same length, no NaN
    //       or infinite values)
    //   y : one label per row, each -1 or +1, and BOTH classes must appear
    // Throws InvalidArgument for empty data, zero features, labels other than
    //        -1 / +1, a single class, NaN / infinite values, or when training
    //        diverges (weights become NaN or infinity).
    // Throws DimensionMismatch if X and y have different sizes or X has rows
    //        of different lengths.
    // If fit() throws, the model is left exactly as it was before the call.
    // Calling fit() again replaces the old model (it re-trains from zero and
    // may use a different number of features).
    void fit(const Matrix& X, const std::vector<int>& y);

    // Score of one sample / many samples (w . x + b). See decision.hpp.
    // Positive = +1 side, negative = -1 side, bigger = more sure.
    // Throws NotFitted if fit() has not succeeded yet.
    // Throws DimensionMismatch for the wrong number of features.
    // An empty matrix gives an empty result.
    double decision_function(const Vector& x) const;
    Vector decision_function(const Matrix& X) const;

    // Predicted label (+1 or -1) for one sample / many samples.
    // This is just the sign of decision_function (a score of 0 gives +1).
    // Throws NotFitted before fit(); DimensionMismatch for wrong feature count;
    // InvalidArgument if the sample contains NaN.
    int predict(const Vector& x) const;
    std::vector<int> predict(const Matrix& X) const;

    // Read-only access to what the model learned. Throw NotFitted before fit().
    //   hyperplane() : the learned weights w and bias b
    //   history()    : loss after each epoch, epochs run, and whether training
    //                  stopped early because it converged
    // The returned references stay valid until the next fit() call.
    const Hyperplane& hyperplane() const;
    const TrainingHistory& history() const;

    // Row numbers (in the training data) of the support vectors, i.e. the
    // training points on or inside the margin (found by find_support_vectors
    // in support_vectors.hpp, with its default tolerance).
    // Throws NotFitted before fit(). May be empty if every training point is
    // comfortably outside the margin.
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