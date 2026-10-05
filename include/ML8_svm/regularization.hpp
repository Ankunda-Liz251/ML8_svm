// =============================================================================
// regularization.hpp  --  L2 regularization and the full SVM objective
// Owner: Trevor (M6)
//
// BACKGROUND (plain English)
//   Minimising only the hinge loss can give a weight vector w with huge values
//   that fits the training data too tightly (overfitting). Regularization adds
//   a PENALTY for a large w, which pushes the model towards a wider margin.
//
//   The full objective we minimise (see project plan, section 1.3) is:
//
//       J(w, b) = (lambda / 2) * ||w||^2  +  (1/n) * SUM max(0, 1 - y_i*(w.x_i + b))
//                 \_____________________/     \__________________________________/
//                      l2_penalty                          hinge_loss
//
//   lambda (>= 0) controls the strength:
//       small lambda -> weak penalty, narrow margin, risk of overfitting
//       large lambda -> strong penalty, wide margin, risk of underfitting
//
//   IMPORTANT: the bias b is NOT penalised. Only w is.
//
// DEPENDENCIES
//   hyperplane.hpp (M4) for Hyperplane, hinge_loss.hpp (M5) for hinge_loss.
// =============================================================================
#pragma once

#include <vector>

#include <ML8_svm/hyperplane.hpp>  // Hyperplane { Vector w; double b; ... }
#include <ML8_svm/types.hpp>

namespace ml8_svm {

// L2 penalty:  (lambda / 2) * ||w||^2
// Throws InvalidArgument if lambda < 0.
// Example: w = {3,4}, lambda = 0.1  ->  0.05 * 25 = 1.25
double l2_penalty(const Vector& w, double lambda);

// Gradient of the L2 penalty with respect to w:  lambda * w
// (This is added to the hinge subgradient gw inside the optimizer.)
// Throws InvalidArgument if lambda < 0.
// Example: w = {3,4}, lambda = 0.1  ->  {0.3, 0.4}
Vector l2_gradient(const Vector& w, double lambda);

// Full objective J = hinge_loss(h, X, y) + l2_penalty(h.w, lambda).
// Throws InvalidArgument (lambda < 0, empty X) or DimensionMismatch (X, y or
// h.w sizes do not fit together).
double objective(const Hyperplane& h, const Matrix& X, const std::vector<int>& y,
                 double lambda);

}  // namespace ml8_svm
