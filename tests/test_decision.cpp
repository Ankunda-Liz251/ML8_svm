
//  test_decision.cpp     
//  Tests for decision.hpp. Expected numbers were worked out BY HAND.
//  Test line:  w = {2, -1},  b = 0.5      so   score(x) = 2*x1 - x2 + 0.5
//
//  Uses test_assert() from tests_utils.hpp (M3). A failed test_assert throws,
//  so the program stops with a non-zero exit code and ctest reports a failure.
#include <ML8_svm/decision.hpp>
#include <ML8_svm/errors.hpp>
#include "tests_utils.hpp"

#include <cmath>
#include <limits>

using namespace ml8_svm;

// True when a and b differ by at most eps (for comparing decimal numbers).
static bool almost_equal(double a, double b, double eps = 1e-12) {
    return std::fabs(a - b) <= eps;
}

// Passes ONLY if the code throws exactly the expected exception type.
// (The shared expect_throw in tests_utils.hpp also passes when nothing is
//  thrown, so this local helper is safer. See the note in the Week 2 report.)
template <typename ExceptionType, typename Function>
static void expect_throw_type(Function function, const std::string& message) {
    bool right_type = false;
    try { function(); }
    catch (const ExceptionType&) { right_type = true; }
    catch (...) {}
    test_assert(right_type, message);
}

int main() {
    Hyperplane h;
    h.w = {2.0, -1.0};
    h.b = 0.5;

    // --- one sample ---------------------------------------------------------
    test_assert(almost_equal(decision_function(h, Vector{1.0, 1.0}), 1.5),
                "score of (1,1) is 2-1+0.5 = 1.5");
    test_assert(almost_equal(decision_function(h, Vector{0.0, 3.0}), -2.5),
                "score of (0,3) is 0-3+0.5 = -2.5");
    test_assert(almost_equal(decision_function(h, Vector{0.0, 0.0}), 0.5),
                "score of (0,0) is just the bias 0.5");

    // --- many samples: must equal the one-by-one results ---------------------
    Matrix X = {{1.0, 1.0}, {0.0, 3.0}, {0.0, 0.0}};
    Vector scores = decision_function(h, X);
    test_assert(scores.size() == 3, "matrix version returns one score per row");
    test_assert(almost_equal(scores[0], 1.5), "matrix score 0 is 1.5");
    test_assert(almost_equal(scores[1], -2.5), "matrix score 1 is -2.5");
    test_assert(almost_equal(scores[2], 0.5), "matrix score 2 is 0.5");

    // --- empty matrix gives an empty result ----------------------------------
    test_assert(decision_function(h, Matrix{}).empty(), "empty matrix gives empty result");

    // --- wrong number of features --------------------------------------------
    expect_throw_type<DimensionMismatch>(
        [&] { decision_function(h, Vector{1.0}); },
        "too few features throws DimensionMismatch");
    expect_throw_type<DimensionMismatch>(
        [&] { decision_function(h, Vector{1.0, 2.0, 3.0}); },
        "too many features throws DimensionMismatch");
    expect_throw_type<DimensionMismatch>(
        [&] { decision_function(h, Matrix{{1.0, 1.0}, {1.0}}); },
        "a short row in a matrix throws DimensionMismatch");

    // --- to_label --------------------------------------------------------------
    test_assert(to_label(0.0001) == +1, "small positive score gives +1");
    test_assert(to_label(-0.0001) == -1, "small negative score gives -1");
    test_assert(to_label(0.0) == +1, "score 0 goes to +1 (boundary rule)");
    test_assert(to_label(-0.0) == +1, "negative zero is still 0, so +1");
    test_assert(to_label(1e9) == +1, "huge positive score gives +1");
    test_assert(to_label(-1e9) == -1, "huge negative score gives -1");
    test_assert(to_label(std::numeric_limits<double>::infinity()) == +1, "+infinity gives +1");
    test_assert(to_label(-std::numeric_limits<double>::infinity()) == -1, "-infinity gives -1");
    expect_throw_type<InvalidArgument>(
        [&] { to_label(std::nan("")); },
        "NaN score throws InvalidArgument");

    // --- NaN in the input leads to an error, not a silent wrong label --------
    expect_throw_type<InvalidArgument>(
        [&] { to_label(decision_function(h, Vector{std::nan(""), 1.0})); },
        "NaN feature leads to InvalidArgument, not a wrong label");

    return 0;   // reached only if every test_assert passed
}