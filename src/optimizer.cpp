
#include <ML8_svm/optimizer.hpp>
#include <ML8_svm/errors.hpp>
#include <ML8_svm/linalg.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace ml8_svm {

TrainingHistory optimize(
    Hyperplane& hyperplane,
    const Matrix& X,
    const std::vector<int>& y,
    double lambda,
    const OptimizerConfig& config
) {
    check_X_y(X, y, "optimize");

    if (lambda < 0.0 || !std::isfinite(lambda)) {
        throw InvalidArgument("lambda must be finite and >= 0");
    }

    if (config.learning_rate <= 0.0 ||
        !std::isfinite(config.learning_rate)) {
        throw InvalidArgument("learning_rate must be finite and > 0");
    }

    if (config.batch_size == 0) {
        throw InvalidArgument("batch_size must be > 0");
    }

    if (!std::isfinite(config.tol) || config.tol < 0.0) {
        throw InvalidArgument("tol must be finite and >= 0");
    }

    const std::size_t sample_count = X.size();
    const std::size_t feature_count = X[0].size();
    const std::size_t batch_size =
        std::min(config.batch_size, sample_count);

    for (int label : y) {
        if (label != -1 && label != 1) {
            throw InvalidArgument("SVM labels must be -1 or +1");
        }
    }

    if (hyperplane.w.empty()) {
        hyperplane.w.assign(feature_count, 0.0);
    } else if (hyperplane.w.size() != feature_count) {
        throw DimensionMismatch(
            "Hyperplane weights must match the number of features"
        );
    }

    if (!std::isfinite(hyperplane.b)) {
        throw InvalidArgument("Hyperplane bias must be finite");
    }

    for (double weight : hyperplane.w) {
        if (!std::isfinite(weight)) {
            throw InvalidArgument("Hyperplane weights must be finite");
        }
    }

    // Calculate the regularized hinge-loss objective.
    const auto calculate_loss = [&]() {
        double weight_squared_sum = 0.0;

        for (double weight : hyperplane.w) {
            weight_squared_sum += weight * weight;
        }

        double hinge_sum = 0.0;

        for (std::size_t i = 0; i < sample_count; ++i) {
            const double margin =
                static_cast<double>(y[i]) * hyperplane.score(X[i]);

            hinge_sum += std::max(0.0, 1.0 - margin);
        }

        return 0.5 * lambda * weight_squared_sum
             + hinge_sum / static_cast<double>(sample_count);
    };

    TrainingHistory history;

    if (config.epochs == 0) {
        history.loss.push_back(calculate_loss());
        return history;
    }

    std::vector<std::size_t> indices(sample_count);
    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937 generator(config.seed);

    double previous_loss = calculate_loss();

    history.loss.push_back(previous_loss);

    for (std::size_t epoch = 0; epoch < config.epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), generator);

        const double learning_rate = config.decay
            ? config.learning_rate /
                  std::sqrt(static_cast<double>(epoch + 1))
            : config.learning_rate;

        for (std::size_t start = 0;
             start < sample_count;
             start += batch_size) {

            const std::size_t end =
                std::min(start + batch_size, sample_count);

            Vector gradient_w(feature_count, 0.0);
            double gradient_b = 0.0;

            for (std::size_t j = start; j < end; ++j) {
                const std::size_t i = indices[j];

                const double margin =
                    static_cast<double>(y[i]) * hyperplane.score(X[i]);

                // Hinge loss contributes a gradient when margin < 1.
                if (margin < 1.0) {
                    const double label = static_cast<double>(y[i]);

                    for (std::size_t k = 0; k < feature_count; ++k) {
                        gradient_w[k] -= label * X[i][k];
                    }

                    gradient_b -= label;
                }
            }

            const double inverse_batch_size =
                1.0 / static_cast<double>(end - start);

            for (std::size_t k = 0; k < feature_count; ++k) {
                gradient_w[k] =
                    lambda * hyperplane.w[k]
                    + gradient_w[k] * inverse_batch_size;

                hyperplane.w[k] -= learning_rate * gradient_w[k];
            }

            gradient_b *= inverse_batch_size;
            hyperplane.b -= learning_rate * gradient_b;
        }

        const double current_loss = calculate_loss();
        history.loss.push_back(current_loss);
        history.epochs_run = epoch + 1;

        if (config.tol > 0.0 &&
            std::abs(previous_loss - current_loss) < config.tol) {
            break;
        }

        previous_loss = current_loss;
    }

    return history;
}

} // namespace ml8_svm
