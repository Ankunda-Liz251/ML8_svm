#include <ML8_svm/hinge_loss.hpp>
#include <ML8_svm/errors.hpp>
#include <ML8_svm/hyperplane.hpp>
#include <ML8_svm/types.hpp>

#include <cmath>
#include <cstddef>
namespace ml8_svm {
namespace {
void validate_hinge_inputs(const Hyperplane& model,
                           const Matrix& X,
                           const std::vector<int>& y) {
    if (X.empty()) {
        throw InvalidArgument("hinge loss requires at least one sample");
    }
    if (X.size() != y.size()) {
        throw DimensionMismatch("number of samples must match number of labels");
    }
    if (model.w.empty()) {
        throw InvalidArgument("hyperplane weight vector must not be empty");
    }

    for (std::size_t i = 0; i < X.size(); ++i) {
        if (X[i].size() != model.w.size()) {
            throw DimensionMismatch("sample feature count must match weight count");
        }
        if (y[i] != -1 && y[i] != 1) {
            throw InvalidArgument("labels must be encoded as -1 or +1");
        }
    }
}
} // namespace

double hinge_loss(const Hyperplane& model,
                  const Matrix& X,
                  const std::vector<int>& y) {
    validate_hinge_inputs(model, X, y);

    double total = 0.0;
    for (std::size_t i = 0; i < X.size(); ++i) {
        const double margin = static_cast<double>(y[i]) * model.score(X[i]);
        total += (margin < 1.0) ? (1.0 - margin) : 0.0;
    }

    return total / static_cast<double>(X.size());
}

void hinge_subgradient(const Hyperplane& model,
                       const Matrix& X,
                       const std::vector<int>& y,
                       Vector& gw,
                       double& gb) {
    validate_hinge_inputs(model, X, y);

    gw.assign(model.w.size(), 0.0);
    gb = 0.0;
    const double inverse_n = 1.0 / static_cast<double>(X.size());

    for (std::size_t i = 0; i < X.size(); ++i) {
        const double margin = static_cast<double>(y[i]) * model.score(X[i]);

        // At the hinge kink (margin == 1), choose the valid zero subgradient.
        if (margin < 1.0) {
            const double factor = -static_cast<double>(y[i]) * inverse_n;
            for (std::size_t j = 0; j < model.w.size(); ++j) {
                gw[j] += factor * X[i][j];
            }
            gb += factor;
        }
    }
}

} // namespace ml8_svm

