
#include <ML8_svm/optimizer.hpp>
#include <ML8_svm/hinge_loss.hpp>
#include <ML8_svm/regularization.hpp>
#include <ML8_svm/errors.hpp>
#include <ML8_svm/linalg.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <string>

namespace ml8_svm {

TrainingHistory optimize(
    Hyperplane& hyperplane,
    const Matrix& X,
    const std::vector<int>& y,
    double lambda,
    const OptimizerConfig& config
) {
    // Validate configuration and training data.
    if (X.empty() || X.size() != y.size()) {
        throw InvalidArgument("X and y must contain the same non-zero number of samples.");
    }

    if (lambda < 0.0 || !std::isfinite(lambda)) {
        throw InvalidArgument("lambda must be finite and non-negative.");
    }

    if (config.learning_rate <= 0.0 ||
        !std::isfinite(config.learning_rate)) {
        throw InvalidArgument("learning_rate must be finite and positive.");
    }

    if (config.epochs == 0 || config.batch_size == 0) {
        throw InvalidArgument("epochs and batch_size must be positive.");
    }

    if (config.tol < 0.0 || !std::isfinite(config.tol)) {
        throw InvalidArgument("tol must be finite and non-negative.");
    }

    const std::size_t n = X.size();
    const std::size_t d = X.front().size();

    if (d == 0 || hyperplane.w.size() != d) {
        throw DimensionMismatch("Feature dimensions must match hyperplane weights.");
    }

    for (std::size_t i = 0; i < n; ++i) {
        if (X[i].size() != d) {
            throw DimensionMismatch("All samples must have the same feature count.");
        }

        if (y[i] != -1 && y[i] != 1) {
            throw InvalidArgument("Labels must be -1 or +1.");
        }
    }

    TrainingHistory history;
    history.loss.reserve(config.epochs);

    std::vector<std::size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    // A fixed seed makes the sample order reproducible.
    std::mt19937 generator(config.seed);

    double previous_loss = objective(hyperplane, X, y, lambda);

    for (std::size_t epoch = 0; epoch < config.epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), generator);

        // Optional learning-rate decay.
        const double learning_rate = config.decay
            ? config.learning_rate / (1.0 + static_cast<double>(epoch))
            : config.learning_rate;

        // Process mini-batches. batch_size = 1 gives SGD;
        // batch_size >= n gives batch gradient descent.
        for (std::size_t start = 0; start < n;
             start += config.batch_size) {
            const std::size_t end =
                std::min(start + config.batch_size, n);

            Matrix batch_X;
            std::vector<int> batch_y;
            batch_X.reserve(end - start);
            batch_y.reserve(end - start);

            for (std::size_t k = start; k < end; ++k) {
                batch_X.push_back(X[indices[k]]);
                batch_y.push_back(y[indices[k]]);
            }

            Vector gw(d, 0.0);
            double gb = 0.0;

            hinge_subgradient(hyperplane, batch_X, batch_y, gw, gb);

            // Add L2 regularization to the weight gradient only.
            const Vector reg_gradient = l2_gradient(hyperplane.w, lambda);

            for (std::size_t j = 0; j < d; ++j) {
                gw[j] += reg_gradient[j];
                hyperplane.w[j] -= learning_rate * gw[j];
            }

            // The bias is not regularized.
            hyperplane.b -= learning_rate * gb;
        }

        const double current_loss = objective(hyperplane, X, y, lambda);
        history.loss.push_back(current_loss);
        history.epochs_run = epoch + 1;

        // Stop if improvement is no greater than the tolerance.
        if (std::abs(previous_loss - current_loss) <= config.tol) {
            history.converged = true;
            break;
        }

        previous_loss = current_loss;
    }

    return history;
}

} // namespace ml8_svm
