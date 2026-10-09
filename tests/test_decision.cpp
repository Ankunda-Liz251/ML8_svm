// =============================================================================
//  test_decision.cpp     Owner: Liz (M9)
//  Tests for decision.hpp. Expected numbers were worked out BY HAND.
//  Test line:  w = {2, -1},  b = 0.5      so   score(x) = 2*x1 - x2 + 0.5
// =============================================================================
#include <ML8_svm/decision.hpp>
#include <ML8_svm/errors.hpp>
#include "test_utils.hpp"

#include <cmath>
#include <limits>

using namespace ml8_svm;

int main() {
    Hyperplane h;
    h.w = {2.0, -1.0};
    h.b = 0.5;

    // --- one sample ---------------------------------------------------------
    TEST_ASSERT_NEAR(decision_function(h, Vector{1.0, 1.0}), 1.5, 1e-12);   // 2-1+0.5
    TEST_ASSERT_NEAR(decision_function(h, Vector{0.0, 3.0}), -2.5, 1e-12);  // 0-3+0.5
    TEST_ASSERT_NEAR(decision_function(h, Vector{0.0, 0.0}), 0.5, 1e-12);   // bias only

    // --- many samples: must equal the one-by-one results ---------------------
    Matrix X = {{1.0, 1.0}, {0.0, 3.0}, {0.0, 0.0}};
    Vector scores = decision_function(h, X);
    TEST_ASSERT(scores.size() == 3);
    TEST_ASSERT_NEAR(scores[0], 1.5, 1e-12);
    TEST_ASSERT_NEAR(scores[1], -2.5, 1e-12);
    TEST_ASSERT_NEAR(scores[2], 0.5, 1e-12);

    // --- empty matrix gives an empty result ----------------------------------
    TEST_ASSERT(decision_function(h, Matrix{}).empty());

    // --- wrong number of features --------------------------------------------
    TEST_EXPECT_THROWS(decision_function(h, Vector{1.0}), DimensionMismatch);
    TEST_EXPECT_THROWS(decision_function(h, Vector{1.0, 2.0, 3.0}), DimensionMismatch);
    TEST_EXPECT_THROWS(decision_function(h, Matrix{{1.0, 1.0}, {1.0}}), DimensionMismatch);

    // --- to_label --------------------------------------------------------------
    TEST_ASSERT(to_label(0.0001) == +1);
    TEST_ASSERT(to_label(-0.0001) == -1);
    TEST_ASSERT(to_label(0.0) == +1);       // boundary rule: 0 goes to +1
    TEST_ASSERT(to_label(-0.0) == +1);
    TEST_ASSERT(to_label(1e9) == +1);
    TEST_ASSERT(to_label(-1e9) == -1);
    TEST_ASSERT(to_label(std::numeric_limits<double>::infinity()) == +1);
    TEST_ASSERT(to_label(-std::numeric_limits<double>::infinity()) == -1);
    TEST_EXPECT_THROWS(to_label(std::nan("")), InvalidArgument);

    // --- NaN in the input leads to an error, not a silent wrong label --------
    TEST_EXPECT_THROWS(to_label(decision_function(h, Vector{std::nan(""), 1.0})), InvalidArgument);

    TEST_FINISH();
}