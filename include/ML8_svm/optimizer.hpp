
#ifndef ML8_SVM_OPTIMIZER_HPP
#define ML8_SVM_OPTIMIZER_HPP

#include <cstddef>
#include <vector>

#include <ML8_svm/types.hpp>
#include <ML8_svm/hyperplane.hpp>

namespace ml8_svm {

/**
 * @brief Configuration settings for gradient-based SVM optimization.
 */
struct OptimizerConfig {
    double learning_rate{0.01};
    std::size_t epochs{100};
    std::size_t batch_size{32};
    double tol{1e-4};
    bool decay{false};
    unsigned seed{42};
};

/**
 * @brief Summary of the training process.
 */
struct TrainingHistory {
    std::vector<double> loss; ///< Objective loss after each completed epoch.
    size_t epochs_run{0};     ///< Number of epochs actually completed.
    bool converged{false};    ///< True if the stopping tolerance was reached.
};

/**
 * @brief Train a linear SVM using gradient-based optimization.
 *
 * Supports batch, stochastic, and mini-batch updates. The batch mode is
 * selected by using a batch size at least as large as the training set;
 * batch_size = 1 selects stochastic gradient descent.
 *
 * @param hyperplane Model whose weights and bias are updated in place.
 * @param X Training samples, with one feature vector per sample.
 * @param y Binary labels; each label must be -1 or +1.
 * @param lambda Non-negative L2 regularization strength.
 * @param config Optimizer settings, including learning rate and epoch limit.
 * @return Loss history, number of epochs run, and stopping status.
 *
 * @throws InvalidArgument If the data, labels, or configuration are invalid.
 * @throws DimensionMismatch If feature dimensions are inconsistent.
 */
TrainingHistory optimize(
    Hyperplane& hyperplane,
    const Matrix& X,
    const std::vector<int>& y,
    double lambda,
    const OptimizerConfig& config
);

} // namespace ml8_svm

#endif // ML8_SVM_OPTIMIZER_HPP
