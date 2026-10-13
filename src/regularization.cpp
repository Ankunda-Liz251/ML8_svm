// L2 regularization and the full SVM objective (declarations in regularization.hpp).
#include <ML8_svm/regularization.hpp>

#include <ML8_svm/errors.hpp>
#include <ML8_svm/hinge_loss.hpp>
#include <ML8_svm/linalg.hpp>

namespace ml8_svm {

namespace {

// lambda must be a non-negative number. Written as !(lambda >= 0) so that
// NaN is rejected as well, since every comparison with NaN is false.
void check_lambda(double lambda, const std::string& where) {
    if (!(lambda >= 0.0)) {
        throw InvalidArgument(where + ": lambda must be >= 0");
    }
}

}  // namespace

double l2_penalty(const Vector& w, double lambda) {
    check_lambda(lambda, "l2_penalty");
    return 0.5 * lambda * dot(w, w);  // (lambda / 2) * ||w||^2
}

Vector l2_gradient(const Vector& w, double lambda) {
    check_lambda(lambda, "l2_gradient");
    return scale(w, lambda);
}

double objective(const Hyperplane& h, const Matrix& X, const std::vector<int>& y,
                 double lambda) {
    check_lambda(lambda, "objective");
    check_X_y(X, y, "objective");
    check_same_size(h.w, X[0], "objective");  // weights must match the feature count
    return hinge_loss(h, X, y) + l2_penalty(h.w, lambda);
}

}  // namespace ml8_svm
