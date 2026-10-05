// =============================================================================
// regularization.cpp  --  WEEK 1 STUB (implemented in Week 2)
// Owner: Trevor (M6)
//
// A "stub" is a function that compiles but is not finished yet. It lets
// teammates (M7 optimizer, M9 LinearSVM) build and link their code today.
// Each stub throws, so nobody mistakes it for a real result.
// =============================================================================
#include <ML8_svm/regularization.hpp>

#include <stdexcept>

namespace ml8_svm {

double l2_penalty(const Vector& /*w*/, double /*lambda*/) {
    // TODO (Week 2): validate lambda >= 0, return 0.5 * lambda * dot(w, w).
    throw std::logic_error("l2_penalty: not implemented yet (Week 2)");
}

Vector l2_gradient(const Vector& /*w*/, double /*lambda*/) {
    // TODO (Week 2): validate lambda >= 0, return scale(w, lambda).
    throw std::logic_error("l2_gradient: not implemented yet (Week 2)");
}

double objective(const Hyperplane& /*h*/, const Matrix& /*X*/,
                 const std::vector<int>& /*y*/, double /*lambda*/) {
    // TODO (Week 2): check_X_y, validate lambda, then
    //   return hinge_loss(h, X, y) + l2_penalty(h.w, lambda);
    throw std::logic_error("objective: not implemented yet (Week 2)");
}

}  // namespace ml8_svm
