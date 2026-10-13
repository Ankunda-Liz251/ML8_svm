// Unit tests for regularization.hpp.
#include <ML8_svm/errors.hpp>
#include <ML8_svm/regularization.hpp>

#include "test_utils.hpp"

using namespace ml8_svm;

int main() {
    // l2_penalty: w = (3, 4) has ||w||^2 = 25, so lambda = 0.1 gives 0.05 * 25.
    ASSERT_NEAR(l2_penalty({3, 4}, 0.1), 1.25, 1e-12);
    ASSERT_NEAR(l2_penalty({3, 4}, 0.0), 0.0, 1e-12);
    ASSERT_NEAR(l2_penalty({0, 0}, 5.0), 0.0, 1e-12);
    EXPECT_THROWS(l2_penalty({3, 4}, -0.1), InvalidArgument);

    // l2_gradient is lambda * w.
    Vector g = l2_gradient({3, 4}, 0.1);
    ASSERT_NEAR(g[0], 0.3, 1e-12);
    ASSERT_NEAR(g[1], 0.4, 1e-12);
    ASSERT_TRUE(l2_gradient({3, 4}, 0.0) == Vector({0, 0}));
    EXPECT_THROWS(l2_gradient({3, 4}, -1.0), InvalidArgument);

    // objective: w = (1, 0), b = 0. Sample scores are 0.5 and 2, so the hinge
    // terms are 0.5 and 0 (mean 0.25). The penalty with lambda = 0.2 is 0.1.
    Hyperplane h;
    h.w = {1, 0};
    h.b = 0.0;
    Matrix X = {{0.5, 0}, {2, 0}};
    std::vector<int> y = {1, 1};
    ASSERT_NEAR(objective(h, X, y, 0.2), 0.35, 1e-12);
    ASSERT_NEAR(objective(h, X, y, 0.0), 0.25, 1e-12);

    // objective rejects bad input.
    EXPECT_THROWS(objective(h, X, y, -0.2), InvalidArgument);
    EXPECT_THROWS(objective(h, {}, {}, 0.2), InvalidArgument);
    EXPECT_THROWS(objective(h, X, {1}, 0.2), DimensionMismatch);
    h.w = {1, 0, 0};
    EXPECT_THROWS(objective(h, X, y, 0.2), DimensionMismatch);

    return 0;
}
