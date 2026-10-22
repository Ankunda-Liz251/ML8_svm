// Unit tests for regularization.hpp: normal cases, edge cases and invalid input.
#include <limits>

#include <ML8_svm/errors.hpp>
#include <ML8_svm/regularization.hpp>

#include "test_utils.hpp"

using namespace std;
using namespace ml8_svm;

int main() {
    const double nan = numeric_limits<double>::quiet_NaN();

    // ---- l2_penalty = (lambda / 2) * ||w||^2 ----
    ASSERT_NEAR(l2_penalty({3, 4}, 0.1), 1.25, 1e-12);     // ||w||^2 = 25
    ASSERT_NEAR(l2_penalty({-3, -4}, 0.1), 1.25, 1e-12);   // sign does not matter
    ASSERT_NEAR(l2_penalty({1, -2, 2}, 2.0), 9.0, 1e-12);  // ||w||^2 = 9
    ASSERT_NEAR(l2_penalty({1, 1}, 1000.0), 1000.0, 1e-9); // large lambda
    ASSERT_NEAR(l2_penalty({3, 4}, 0.0), 0.0, 1e-12);      // lambda = 0
    ASSERT_NEAR(l2_penalty({0, 0}, 5.0), 0.0, 1e-12);      // w = 0
    ASSERT_NEAR(l2_penalty({}, 0.5), 0.0, 1e-12);          // empty w
    EXPECT_THROWS(l2_penalty({3, 4}, -0.1), InvalidArgument);
    EXPECT_THROWS(l2_penalty({3, 4}, nan), InvalidArgument);

    // ---- l2_gradient = lambda * w ----
    Vector w = {3, 4};
    Vector g = l2_gradient(w, 0.1);
    ASSERT_NEAR(g[0], 0.3, 1e-12);
    ASSERT_NEAR(g[1], 0.4, 1e-12);
    ASSERT_TRUE(w == Vector({3, 4}));                       // input is not modified
    ASSERT_TRUE(l2_gradient({3, 4}, 0.0) == Vector({0, 0}));
    ASSERT_TRUE(l2_gradient({}, 0.5).empty());
    EXPECT_THROWS(l2_gradient({3, 4}, -1.0), InvalidArgument);
    EXPECT_THROWS(l2_gradient({3, 4}, nan), InvalidArgument);

    // The gradient must match a numerical derivative of the penalty.
    const double eps = 1e-6;
    for (size_t i = 0; i < w.size(); ++i) {
        Vector up = w, down = w;
        up[i] += eps;
        down[i] -= eps;
        double numeric = (l2_penalty(up, 0.1) - l2_penalty(down, 0.1)) / (2 * eps);
        ASSERT_NEAR(numeric, g[i], 1e-6);
    }

    // ---- objective = hinge loss + l2 penalty ----
    // w = (1, 0), b = 0. Scores are 0.5 and 2, so the hinge terms are 0.5 and 0
    // (mean 0.25). With lambda = 0.2 the penalty is 0.1.
    Hyperplane h;
    h.w = {1, 0};
    h.b = 0.0;
    Matrix X = {{0.5, 0}, {2, 0}};
    vector<int> y = {1, 1};
    ASSERT_NEAR(objective(h, X, y, 0.2), 0.35, 1e-12);
    ASSERT_NEAR(objective(h, X, y, 0.0), 0.25, 1e-12);      // lambda = 0: hinge loss only

    // One misclassified sample: score = -1, so the hinge term is 2. Penalty is 0.1.
    ASSERT_NEAR(objective(h, {{-1, 0}}, {1}, 0.2), 2.1, 1e-12);

    // The bias is not penalised. Both hyperplanes have the same w and zero hinge
    // loss on this data, so they must give the same objective (0.5 * 0.4 * 1).
    Matrix Xs = {{3, 0}, {-3, 0}};
    vector<int> ys = {1, -1};
    Hyperplane h_b0, h_b1;
    h_b0.w = {1, 0};
    h_b0.b = 0.0;
    h_b1.w = {1, 0};
    h_b1.b = 0.5;
    ASSERT_NEAR(objective(h_b0, Xs, ys, 0.4), 0.2, 1e-12);
    ASSERT_NEAR(objective(h_b1, Xs, ys, 0.4), 0.2, 1e-12);

    // ---- objective rejects invalid input ----
    EXPECT_THROWS(objective(h, X, y, -0.2), InvalidArgument);        // negative lambda
    EXPECT_THROWS(objective(h, X, y, nan), InvalidArgument);         // NaN lambda
    EXPECT_THROWS(objective(h, {}, {}, 0.2), InvalidArgument);       // no samples
    EXPECT_THROWS(objective(h, X, {1}, 0.2), DimensionMismatch);     // too few labels
    EXPECT_THROWS(objective(h, X, {1, 1, 1}, 0.2), DimensionMismatch);  // too many labels
    EXPECT_THROWS(objective(h, {{1, 0}, {1}}, {1, 1}, 0.2), DimensionMismatch);  // ragged rows

    Hyperplane h3;  // 3 weights, but the samples have 2 features
    h3.w = {1, 0, 0};
    h3.b = 0.0;
    EXPECT_THROWS(objective(h3, X, y, 0.2), DimensionMismatch);

    Hyperplane h_empty;  // no weights at all
    h_empty.b = 0.0;
    EXPECT_THROWS(objective(h_empty, X, y, 0.2), DimensionMismatch);

    return 0;
}
