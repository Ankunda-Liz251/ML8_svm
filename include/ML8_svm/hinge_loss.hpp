 #pragma once

#include <vector>
#include <ML8_svm/types.hpp>
#include <ML8_svm/hyperplane.hpp>

namespace ml8_svm {

/**
 * Compute the mean hinge loss over a binary classification dataset.
 *
 * Labels are expected to be encoded as -1 or +1.
 *
 * hinge_loss = (1/n) * sum_i max(0, 1 - y_i * score(x_i))
 */
double hinge_loss(
    const Hyperplane& model,
    const Matrix& X,
    const std::vector<int>& y
);

/**
 * Compute the hinge-loss subgradient with respect to w and b.
 *
 * For each sample:
 *   if y_i * score(x_i) < 1:
 *       gw += -y_i * x_i / n
 *       gb += -y_i / n
 *   otherwise:
 *       contributes zero.
 *
 * The regularization gradient is intentionally NOT included here.
 */
void hinge_subgradient(
    const Hyperplane& model,
    const Matrix& X,
    const std::vector<int>& y,
    Vector& gw,
    double& gb
);

} // namespace ml8_svm
