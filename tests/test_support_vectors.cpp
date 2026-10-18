#include <iostream>
#include <vector>

#include <ML8_svm/ML8_svm.hpp>

#include "test_utils.hpp"

using namespace ml8_svm;

// Hand-checked toy set. Hyperplane w = (1, 0), b = 0, so score(x) = x[0].
//   i  x        y   y*score   support vector (tol = 1e-3)?
//   0  (2, 0)   +1   2.0      no  (outside margin)
//   1  (1, 0)   +1   1.0      yes (exactly on margin)
//   2  (0.5, 0) +1   0.5      yes (inside margin)
//   3  (-1, 0)  -1   1.0      yes (exactly on margin)
//   4  (-3, 0)  -1   3.0      no  (outside margin)
//   5  (-0.2,0) +1  -0.2      yes (misclassified)
static Hyperplane toy_plane() {
    Hyperplane h;
    h.w = {1.0, 0.0};
    h.b = 0.0;
    return h;
}

static Matrix toy_X() {
    return {{2.0, 0.0}, {1.0, 0.0}, {0.5, 0.0}, {-1.0, 0.0}, {-3.0, 0.0}, {-0.2, 0.0}};
}

static std::vector<int> toy_y() { return {1, 1, 1, -1, -1, 1}; }

static bool throws_invalid(const Hyperplane& h, const Matrix& X, const std::vector<int>& y,
                           double tol = 1e-3) {
    try {
        find_support_vectors(h, X, y, tol);
    } catch (const InvalidArgument&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

static bool throws_dim(const Hyperplane& h, const Matrix& X, const std::vector<int>& y) {
    try {
        find_support_vectors(h, X, y);
    } catch (const DimensionMismatch&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

static int failures = 0;
#define SV_CHECK(cond)                                                          \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::cerr << "FAILED: " #cond " (line " << __LINE__ << ")\n";       \
            ++failures;                                                         \
        }                                                                       \
    } while (0)

int main() {
    const Hyperplane h = toy_plane();

    // 1. Hand-checked toy set.
    {
        const auto sv = find_support_vectors(h, toy_X(), toy_y(), 1e-3);
        const std::vector<std::size_t> expected = {1, 2, 3, 5};
        SV_CHECK(sv == expected);
    }

    // 2. Tolerance: a point just outside the margin is only included with a larger tol.
    {
        Matrix X = {{1.0005, 0.0}};
        std::vector<int> y = {1};
        SV_CHECK(find_support_vectors(h, X, y, 0.0).empty());
        SV_CHECK(find_support_vectors(h, X, y, 1e-3).size() == 1);
    }

    // 3. A point exactly on the margin is included even with tol = 0.
    {
        Matrix X = {{1.0, 0.0}};
        std::vector<int> y = {1};
        SV_CHECK(find_support_vectors(h, X, y, 0.0).size() == 1);
    }

    // 4. No support vectors when every sample is well outside the margin.
    {
        Matrix X = {{5.0, 0.0}, {-5.0, 0.0}};
        std::vector<int> y = {1, -1};
        SV_CHECK(find_support_vectors(h, X, y).empty());
    }

    // 5. A bias term shifts the margin: score = x0 - 2, so x0 = 3 has y*score = 1.
    {
        Hyperplane hb;
        hb.w = {1.0, 0.0};
        hb.b = -2.0;
        Matrix X = {{3.0, 0.0}, {10.0, 0.0}};
        std::vector<int> y = {1, 1};
        const auto sv = find_support_vectors(hb, X, y, 0.0);
        SV_CHECK(sv.size() == 1 && sv[0] == 0);
    }

    // 6. Invalid inputs.
    SV_CHECK(throws_invalid(h, Matrix{}, std::vector<int>{}));            // empty X
    SV_CHECK(throws_invalid(h, toy_X(), toy_y(), -0.1));                  // negative tol
    SV_CHECK(throws_invalid(h, Matrix{{1.0, 0.0}}, std::vector<int>{0})); // label not +-1
    SV_CHECK(throws_dim(h, toy_X(), std::vector<int>{1, 1}));             // X/y length mismatch
    SV_CHECK(throws_dim(h, Matrix{{1.0, 0.0, 0.0}}, std::vector<int>{1}));// wrong row length

    if (failures == 0) {
        std::cout << "test_support_vectors: all tests passed\n";
        return 0;
    }
    std::cerr << "test_support_vectors: " << failures << " failure(s)\n";
    return 1;
}