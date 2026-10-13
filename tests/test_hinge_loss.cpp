#include <ML8_svm/hinge_loss.hpp>
#include <ML8_svm/hyperplane.hpp>
#include <ML8_svm/types.hpp>

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

namespace {
bool near(double a, double b, double tol = 1e-6) {
    return std::abs(a - b) <= tol;
}

double numerical_derivative_w0(const ml8_svm::Hyperplane& model,
                               const ml8_svm::Matrix& X,
                               const std::vector<int>& y,
                               double eps = 1e-6) {
    ml8_svm::Hyperplane plus = model;
    ml8_svm::Hyperplane minus = model;
    plus.w[0] += eps;
    minus.w[0] -= eps;
    return (ml8_svm::hinge_loss(plus, X, y)
          - ml8_svm::hinge_loss(minus, X, y)) / (2.0 * eps);
}

double numerical_derivative_b(const ml8_svm::Hyperplane& model,
                              const ml8_svm::Matrix& X,
                              const std::vector<int>& y,
                              double eps = 1e-6) {
    ml8_svm::Hyperplane plus = model;
    ml8_svm::Hyperplane minus = model;
    plus.b += eps;
    minus.b -= eps;
    return (ml8_svm::hinge_loss(plus, X, y)
          - ml8_svm::hinge_loss(minus, X, y)) / (2.0 * eps);
}
} // namespace

int main() {
    
    // Planned tests:
     using namespace ml8_svm;

    // Test 1: one correctly classified point with margin greater than 1 -> loss 0.
    {
        Hyperplane h{{2.0}, 0.0};
        Matrix X{{1.0}};
        std::vector<int> y{1};
        assert(near(hinge_loss(h, X, y), 0.0));
    }

    // Test 2: one point with score 0.25 and label +1 -> loss 0.75.
    {
        Hyperplane h{{0.25}, 0.0};
        Matrix X{{1.0}};
        std::vector<int> y{1};
        assert(near(hinge_loss(h, X, y), 0.75));
    }

    // Test 3: average loss across two samples.
    {
        Hyperplane h{{1.0}, 0.0};
        Matrix X{{0.0}, {2.0}};
        std::vector<int> y{1, -1};
        // Margins are 0 and -2; losses are 1 and 3; mean = 2.
        assert(near(hinge_loss(h, X, y), 2.0));
    }

    // Test 4: known subgradient for two active hinge-loss samples.
    {
        Hyperplane h{{0.0, 0.0}, 0.0};
        Matrix X{{2.0, 4.0}, {4.0, 2.0}};
        std::vector<int> y{1, -1};
        Vector gw;
        double gb = 99.0;
        hinge_subgradient(h, X, y, gw, gb);
        assert(gw.size() == 2);
        assert(near(gw[0], 1.0));  // (-2 + 4) / 2
        assert(near(gw[1], -1.0)); // (-4 + 2) / 2
        assert(near(gb, 0.0));     // (-1 + 1) / 2
    }

    // Test 5: at margin == 1, this implementation chooses the zero subgradient.
    {
        Hyperplane h{{1.0}, 0.0};
        Matrix X{{1.0}};
        std::vector<int> y{1};
        Vector gw;
        double gb = 5.0;
        hinge_subgradient(h, X, y, gw, gb);
        assert(near(gw[0], 0.0));
        assert(near(gb, 0.0));
    }

    // Test 6: compare analytic subgradients with numerical gradients away from the kink.
    {
        Hyperplane h{{0.2, -0.1}, 0.05};
        Matrix X{{1.0, 2.0}, {-1.0, 1.0}, {2.0, -1.0}};
        std::vector<int> y{1, -1, 1};
        Vector gw;
        double gb = 0.0;
        hinge_subgradient(h, X, y, gw, gb);

        assert(near(gw[0], numerical_derivative_w0(h, X, y), 1e-5));
        assert(near(gb, numerical_derivative_b(h, X, y), 1e-5));
    }

    std::cout << "All hinge-loss tests passed.\n";
    return 0;
}

 