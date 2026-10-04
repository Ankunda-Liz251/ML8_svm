#ifndef ML8_SVM_OPTIMIZER_HPP
#define ML8_SVM_OPTIMIZER_HPP

#include <vector>
#include <cstddef>
#include <ML8_svm/types.hpp>
#include <ML8_svm/hyperplane.hpp>

namespace ml8_svm {

struct OptimizerConfig {
    double learning_rate{0.01};
    size_t epochs{100};
    size_t batch_size{32};
    double tol{1e-4};
    bool decay{false};
    unsigned seed{42};
};

struct TrainingHistory {
    std::vector<double> loss;
    size_t epochs_run{0};
    bool converged{false};
};

TrainingHistory optimize(
    Hyperplane& hyperplane,
    const Matrix& X,
    const std::vector<int>& y,
    double lambda,
    const OptimizerConfig& config
);

} // namespace ml8_svm

#endif // ML8_SVM_OPTIMIZER_HPP