#ifndef ML8_SVM_SUPPORT_VECTORS_HPP
#define ML8_SVM_SUPPORT_VECTORS_HPP

#include <cstddef>
#include <vector>

#include <ML8_svm/hyperplane.hpp>
#include <ML8_svm/types.hpp>

namespace ml8_svm {

/**
 * Identify the support vectors of a trained linear SVM.
 *
 * A sample i is a support vector when it lies on or inside the margin:
 *     y_i * score(x_i) <= 1 + tol
 * where score(x) = w.x + b and labels are encoded as -1 / +1.
 *
 * @param h    Trained hyperplane (w, b).
 * @param X    Samples, one row per sample.
 * @param y    Labels in {-1, +1}, same length as X.
 * @param tol  Non-negative tolerance added to the margin boundary.
 * @return     Indices (into X) of the support vectors, in increasing order.
 *
 * @throws InvalidArgument      if X is empty, a label is not -1/+1, or tol < 0.
 * @throws DimensionMismatch    if X.size() != y.size() or a row length != h.w.size().
 */
std::vector<std::size_t> find_support_vectors(const Hyperplane& h,
                                              const Matrix& X,
                                              const std::vector<int>& y,
                                              double tol = 1e-3);

}  // namespace ml8_svm

#endif  // ML8_SVM_SUPPORT_VECTORS_HPP