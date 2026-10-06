// Tests for module M4: Hyperplane (hyperplane.hpp)
//
// Week 1: skeleton only. The tiny local helper below keeps this file
// self-contained; when M3 merges tests/test_utils.hpp, swap CHECK for its
// assert / expect-throws macros.
// Week 2: enable the TODO tests (hand-computed expected values).

#include <ML8_svm/hyperplane.hpp>

#include <cmath>
#include <iostream>

using namespace ml8_svm;

static int g_failures = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::cerr << "FAILED: " #cond " (" << __FILE__ << ":"            \
                      << __LINE__ << ")\n";                                  \
            ++g_failures;                                                    \
        }                                                                    \
    } while (0)

static bool near(double a, double b, double eps = 1e-9) {
    return std::fabs(a - b) <= eps;
}

// Passes now: the struct stores w and b.
static void test_construction() {
    Hyperplane h;
    h.w = {3.0, 4.0};
    h.b = -1.0;
    CHECK(h.w.size() == 2);
    CHECK(near(h.b, -1.0));
}

// Hand-computed example for week 2: w = (3,4), b = -1, ||w|| = 5.
//   x = (1,1): score = 3+4-1 = 6, distance = 6/5 = 1.2
//   y = +1: functional margin = 6, geometric margin = 1.2
//   margin_width = 2/5 = 0.4
static void test_values_TODO() {
    // Hyperplane h; h.w = {3.0, 4.0}; h.b = -1.0;
    // Vector x = {1.0, 1.0};
    // CHECK(near(h.score(x), 6.0));
    // CHECK(near(h.distance(x), 1.2));
    // CHECK(near(h.functional_margin(x, +1), 6.0));
    // CHECK(near(h.geometric_margin(x, +1), 1.2));
    // CHECK(near(h.functional_margin(x, -1), -6.0));
    // CHECK(near(h.margin_width(), 0.4));
}

static void test_errors_TODO() {
    // w = 0            -> distance / geometric_margin / margin_width throw InvalidArgument
    // x wrong size     -> score throws DimensionMismatch
    // y not in {-1,+1} -> functional_margin throws InvalidArgument
}

int main() {
    test_construction();
    test_values_TODO();
    test_errors_TODO();

    if (g_failures == 0) {
        std::cout << "test_hyperplane: all tests passed\n";
        return 0;
    }
    std::cerr << "test_hyperplane: " << g_failures << " failure(s)\n";
    return 1;
}