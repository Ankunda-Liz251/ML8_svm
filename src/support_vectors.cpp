#include <ML8_svm/support_vectors.hpp>

#include <ML8_svm/errors.hpp>

namespace ml8_svm {

std::vector<std::size_t> find_support_vectors(const Hyperplane& h,
                                              const Matrix& X,
                                              const std::vector<int>& y,
                                              double tol) {
    if (X.empty()) {
        throw InvalidArgument("find_support_vectors: X is empty");
    }
    if (tol < 0.0) {
        throw InvalidArgument("find_support_vectors: tol must be >= 0");
    }
    if (X.size() != y.size()) {
        throw DimensionMismatch("find_support_vectors: X and y have different lengths");
    }

    // Validate everything first so we never return a partial result.
    for (std::size_t i = 0; i < X.size(); ++i) {
        if (X[i].size() != h.w.size()) {
            throw DimensionMismatch("find_support_vectors: row length differs from w");
        }
        if (y[i] != -1 && y[i] != 1) {
            throw InvalidArgument("find_support_vectors: labels must be -1 or +1");
        }
    }

    std::vector<std::size_t> indices;
    for (std::size_t i = 0; i < X.size(); ++i) {
        // Functional margin y_i * (w.x_i + b); on or inside the margin => support vector.
        const double functional_margin = static_cast<double>(y[i]) * h.score(X[i]);
        if (functional_margin <= 1.0 + tol) {
            indices.push_back(i);
        }
    }
    return indices;
}

}  // namespace ml8_svm