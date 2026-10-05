#include <ML8_svm/hinge_loss.hpp>

namespace ml8_svm {

double hinge_loss(
    const Hyperplane& /*model*/,
    const Matrix& /*X*/,
    const std::vector<int>& /*y*/
) {
    // Week 1 skeleton: implementation is completed in Week 2.
    return 0.0;
}

void hinge_subgradient(
    const Hyperplane& model,
    const Matrix& X,
    const std::vector<int>& y,
    Vector& gw,
    double& gb
) {
    // Week 1 skeleton: implementation is completed in Week 2.
    // Initialise outputs so callers have deterministic values.
    gw.assign(model.w.size(), 0.0);
    gb = 0.0;

    (void)X;
    (void)y;
}

} // namespace ml8_svm
