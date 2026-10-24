
#include <ML8_svm/optimizer.hpp>

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

void test_optimizer(std::size_t batch_size, const char* method) {
    ml8_svm::Hyperplane hyperplane;

    ml8_svm::Matrix X = {
        {-2.0},
        {-1.0},
        {1.0},
        {2.0}
    };

    std::vector<int> y = {-1, -1, 1, 1};

    ml8_svm::OptimizerConfig config;
    config.learning_rate = 0.1;
    config.epochs = 100;
    config.batch_size = batch_size;
    config.tol = 0.0;
    config.seed = 42;

    const double lambda = 0.01;

    const auto history = ml8_svm::optimize(
        hyperplane, X, y, lambda, config
    );

    // Training must update the weights.
    assert(!hyperplane.w.empty());
    assert(std::abs(hyperplane.w[0]) > 0.0);

    // History contains initial loss and one entry per completed epoch.
    assert(history.loss.size() == history.epochs_run + 1);
    assert(history.epochs_run == config.epochs);

    // Training should reduce the loss on this separable dataset.
    assert(history.loss.back() < history.loss.front());

    // The learned model should classify both classes correctly.
    assert(hyperplane.score(X[0]) < 0.0);
    assert(hyperplane.score(X[1]) < 0.0);
    assert(hyperplane.score(X[2]) > 0.0);
    assert(hyperplane.score(X[3]) > 0.0);

    std::cout << method << " test passed. Initial loss: "
              << history.loss.front()
              << ", final loss: " << history.loss.back()
              << '\n';
}

int main() {
    // Batch gradient descent: process all four samples per update.
    test_optimizer(4, "Batch gradient descent");

    // Stochastic gradient descent: process one sample per update.
    test_optimizer(1, "SGD");

    std::cout << "All optimizer tests passed.\n";
    return 0;
}
