
#include <ML8_svm/optimizer.hpp>

#include <cassert>
#include <cmath>
#include <iostream>

int main() {
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
    config.batch_size = 4;
    config.tol = 0.0;

    const double lambda = 0.01;

    const auto history = ml8_svm::optimize(
        hyperplane, X, y, lambda, config
    );

    // Training must update the weights.
    assert(!hyperplane.w.empty());
    assert(std::abs(hyperplane.w[0]) > 0.0);

    // The training history must contain the initial and final losses.
    assert(history.loss.size() == config.epochs + 1);
    assert(history.epochs_run == config.epochs);

    // This simple, separable dataset should have a lower final loss.
    assert(history.loss.back() < history.loss.front());

    // The learned hyperplane should classify both sides correctly.
    assert(hyperplane.score(X[0]) < 0.0);
    assert(hyperplane.score(X[3]) > 0.0);

    std::cout << "Optimizer training test passed." << std::endl;
    return 0;
}
