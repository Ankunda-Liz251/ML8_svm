
#include <ML8_svm/optimizer.hpp>
#include <ML8_svm/hyperplane.hpp>

#include <iostream>
#include <vector>

int main() {
    using namespace ml8_svm;

    Matrix X = {{-2.0}, {-1.0}, {1.0}, {2.0}};
    std::vector<int> y = {-1, -1, 1, 1};

    Hyperplane model;
    model.w = {0.0};
    model.b = 0.0;

    OptimizerConfig config;
    config.learning_rate = 0.1;
    config.epochs = 50;
    config.batch_size = 2;
    config.tol = 0.0;
    config.decay = true;
    config.seed = 42;

    TrainingHistory history = optimize(model, X, y, 0.01, config);

    std::cout << "Epoch, Loss\n";
    for (std::size_t i = 0; i < history.loss.size(); ++i) {
        std::cout << i + 1 << ", " << history.loss[i] << '\n';
    }

    std::cout << "\nEpochs run: " << history.epochs_run << '\n';
    std::cout << "Converged: "
              << (history.converged ? "Yes" : "No") << '\n';

    return 0;
}
