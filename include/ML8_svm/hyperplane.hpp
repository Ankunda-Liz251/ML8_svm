#pragma once

/**
 * @file hyperplane.hpp
 * @brief Hyperplane representation and margin computation (module M4).
 *
 * A hyperplane is described by a weight vector w and a bias b:
 *     score(x) = w . x + b
 * The decision boundary is the set of points where score(x) = 0.
 * Labels are encoded as -1 / +1.
 */

#include <ML8_svm/errors.hpp>
#include <ML8_svm/linalg.hpp>
#include <ML8_svm/types.hpp>

namespace ml8_svm {

struct Hyperplane {
    Vector w;       ///< weight vector (normal to the hyperplane)
    double b = 0.0; ///< bias (intercept)

    /// Signed raw score: w . x + b.
    /// @throws DimensionMismatch if x.size() != w.size().
    double score(const Vector& x) const;

    /// Signed geometric distance of x from the hyperplane: score(x) / ||w||.
    /// @throws InvalidArgument if w is the zero vector.
    /// @throws DimensionMismatch if x.size() != w.size().
    double distance(const Vector& x) const;

    /// Functional margin: y * score(x), with y in {-1, +1}.
    /// @throws InvalidArgument if y is not -1 or +1.
    double functional_margin(const Vector& x, int y) const;

    /// Geometric margin: y * score(x) / ||w||, with y in {-1, +1}.
    /// @throws InvalidArgument if w is zero or y is not -1 or +1.
    double geometric_margin(const Vector& x, int y) const;

    /// Width of the margin band: 2 / ||w||.
    /// @throws InvalidArgument if w is the zero vector.
    double margin_width() const;
};

} // namespace ml8_svm