
#include <ML8_svm/optimizer.hpp>
#include <ML8_svm/regularization.hpp>

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace ml8_svm;

Matrix training_data() {
    return {{-2.0}, {-1.0}, {1.0}, {2.0}};
}

std::vector<int> training_labels() {
    return {-1, -1, 1, 1};
}

void test_training(std::size_t batch_size, bool decay) {
    Hyperplane model;
    model.w = {0.0};

    OptimizerConfig config;
    config.learning_rate = 0.1;
    config.epochs = 100;
    config.batch_size = batch_size;
    config.tol = 0.0;
    config.decay = decay;
    config.seed = 42;

    const auto X = training_data();
    const auto y = training_labels();

    const auto history = optimize(model, X, y, 0.01, config);

    assert(!history.loss.empty());
    assert(history.loss.size() == history.epochs_run);
    assert(history.epochs_run <= config.epochs);
    assert(history.loss.back() < history.loss.front());

    assert(model.score(X[0]) < 0.0);
    assert(model.score(X[1]) < 0.0);
    assert(model.score(X[2]) > 0.0);
    assert(model.score(X[3]) > 0.0);

    std::cout << "Training test passed: batch_size=" << batch_size
              << ", decay=" << decay
              << ", epochs=" << history.epochs_run << '\n';
}

void test_early_stopping() {
    Hyperplane model;
    model.w = {0.0};

    OptimizerConfig config;
    config.learning_rate = 0.1;
    config.epochs = 100;
    config.batch_size = 4;
    config.tol = 1e9; // Deliberately large to trigger early stopping.
    config.seed = 42;

    const auto history = optimize(
        model, training_data(), training_labels(), 0.01, config
    );

    assert(history.converged);
    assert(history.epochs_run < config.epochs);
    assert(history.loss.size() == history.epochs_run);

    std::cout << "Early-stopping test passed.\n";
}

void test_reproducibility() {
    Hyperplane first;
    first.w = {0.0};

    Hyperplane second;
    second.w = {0.0};

    OptimizerConfig config;
    config.learning_rate = 0.1;
    config.epochs = 20;
    config.batch_size = 2;
    config.tol = 0.0;
    config.decay = true;
    config.seed = 42;

    const auto X = training_data();
    const auto y = training_labels();

    const auto a = optimize(first, X, y, 0.01, config);
    const auto b = optimize(second, X, y, 0.01, config);

    assert(first.w == second.w);
    assert(first.b == second.b);
    assert(a.loss == b.loss);

    std::cout << "Reproducibility test passed.\n";
}

int main() {
    test_training(4, false); // Batch gradient descent
    test_training(1, false); // Stochastic gradient descent
    test_training(2, true);  // Mini-batch with learning-rate decay
    test_early_stopping();
    test_reproducibility();

    std::cout << "All optimizer tests passed.\n";
    return 0;
}

