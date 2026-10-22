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

#define CHECK_THROWS(expr, ExType)                                           \
    do {                                                                     \
        bool caught = false;                                                 \
        try { (void)(expr); } catch (const ExType&) { caught = true; }       \
        catch (...) {}                                                       \
        if (!caught) {                                                       \
            std::cerr << "FAILED: expected " #ExType " from " #expr          \
                      << " (" << __FILE__ << ":" << __LINE__ << ")\n";       \
            ++g_failures;                                                    \
        }                                                                    \
    } while (0)

static bool near(double a, double b, double eps = 1e-9) {
    return std::fabs(a - b) <= eps;
}

// w = (3,4), b = -1, ||w|| = 5
static Hyperplane make_h() {
    Hyperplane h;
    h.w = {3.0, 4.0};
    h.b = -1.0;
    return h;
}

static void test_construction() {
    Hyperplane h = make_h();
    CHECK(h.w.size() == 2);
    CHECK(near(h.b, -1.0));
}

static void test_score() {
    Hyperplane h = make_h();
    CHECK(near(h.score({1.0, 1.0}), 6.0));    // 3+4-1
    CHECK(near(h.score({0.0, 0.0}), -1.0));   // = b
    CHECK(near(h.score({1.0, -1.0}), -2.0));  // 3-4-1
    CHECK(near(h.score({1.0, 0.0}), 2.0));    // 3-1 -> boundary-ish side check
}

static void test_distance() {
    Hyperplane h = make_h();
    CHECK(near(h.distance({1.0, 1.0}), 1.2));    // 6/5
    CHECK(near(h.distance({1.0, -1.0}), -0.4));  // -2/5
    CHECK(near(h.distance({0.2, 0.1}), 0.0));    // 0.6+0.4-1 = 0, on the boundary
}

static void test_functional_margin() {
    Hyperplane h = make_h();
    CHECK(near(h.functional_margin({1.0, 1.0}, +1), 6.0));
    CHECK(near(h.functional_margin({1.0, 1.0}, -1), -6.0));   // misclassified
    CHECK(near(h.functional_margin({1.0, -1.0}, -1), 2.0));   // correct, negative side
}

static void test_geometric_margin() {
    Hyperplane h = make_h();
    CHECK(near(h.geometric_margin({1.0, 1.0}, +1), 1.2));
    CHECK(near(h.geometric_margin({1.0, 1.0}, -1), -1.2));
    CHECK(near(h.geometric_margin({1.0, -1.0}, -1), 0.4));
}

static void test_scale_invariance() {
    // Doubling (w,b) doubles the functional margin but not the geometric margin.
    Hyperplane h = make_h();
    Hyperplane h2;
    h2.w = {6.0, 8.0};
    h2.b = -2.0;
    Vector x = {1.0, 1.0};
    CHECK(near(h2.functional_margin(x, +1), 2.0 * h.functional_margin(x, +1)));
    CHECK(near(h2.geometric_margin(x, +1), h.geometric_margin(x, +1)));
}

static void test_margin_width() {
    Hyperplane h = make_h();
    CHECK(near(h.margin_width(), 0.4));  // 2/5
    Hyperplane u;
    u.w = {1.0, 0.0};
    CHECK(near(u.margin_width(), 2.0));
}

static void test_errors() {
    Hyperplane h = make_h();
    CHECK_THROWS(h.score({1.0}), DimensionMismatch);
    CHECK_THROWS(h.score({1.0, 2.0, 3.0}), DimensionMismatch);
    CHECK_THROWS(h.functional_margin({1.0, 1.0}, 0), InvalidArgument);
    CHECK_THROWS(h.functional_margin({1.0, 1.0}, 2), InvalidArgument);
    CHECK_THROWS(h.geometric_margin({1.0, 1.0}, 0), InvalidArgument);
}

static void test_zero_weight() {
    Hyperplane z;
    z.w = {0.0, 0.0};
    z.b = 0.5;
    CHECK(near(z.score({3.0, 4.0}), 0.5));              // score still defined
    CHECK(near(z.functional_margin({3.0, 4.0}, -1), -0.5));
    CHECK_THROWS(z.distance({1.0, 1.0}), InvalidArgument);
    CHECK_THROWS(z.geometric_margin({1.0, 1.0}, +1), InvalidArgument);
    CHECK_THROWS(z.margin_width(), InvalidArgument);
}

static void test_boundary_and_sign() {
    Hyperplane h = make_h();
    Vector on = {0.2, 0.1};                       // score = 0
    CHECK(near(h.score(on), 0.0));
    CHECK(near(h.geometric_margin(on, +1), 0.0));
    CHECK(near(h.geometric_margin(on, -1), 0.0));
    // Flipping the label flips the sign of both margins.
    Vector x = {1.0, 1.0};
    CHECK(near(h.functional_margin(x, +1), -h.functional_margin(x, -1)));
    CHECK(near(h.geometric_margin(x, +1), -h.geometric_margin(x, -1)));
}

static void test_support_vector_margin_points() {
    // Points with score = +1 / -1 lie on the margin edges: distance = +/- 1/||w||.
    Hyperplane h = make_h();
    Vector pos = {0.4, 0.2};                      // 1.2+0.8-1 = 1
    Vector neg = {0.0, 0.0};                      // -1
    CHECK(near(h.score(pos), 1.0));
    CHECK(near(h.score(neg), -1.0));
    CHECK(near(h.distance(pos) - h.distance(neg), h.margin_width()));
}

static void test_empty_and_tiny_weights() {
    Hyperplane e;                                  // w empty, b = 0
    CHECK(near(e.score({}), 0.0));
    CHECK_THROWS(e.margin_width(), InvalidArgument);
    CHECK_THROWS(e.distance({}), InvalidArgument);

    Hyperplane t;                                  // tiny but valid weights
    t.w = {1e-6, 0.0};
    CHECK(near(t.margin_width(), 2e6, 1e-3));
}

static void test_large_values() {
    Hyperplane h;
    h.w = {3e8, 4e8};
    h.b = 0.0;
    CHECK(near(h.distance({1.0, 1.0}), 1.4, 1e-9)); // 7e8 / 5e8
}

int main() {
    test_construction();
    test_score();
    test_distance();
    test_functional_margin();
    test_geometric_margin();
    test_scale_invariance();
    test_margin_width();
    test_errors();
    test_zero_weight();
    test_boundary_and_sign();
    test_support_vector_margin_points();
    test_empty_and_tiny_weights();
    test_large_values();

    if (g_failures == 0) {
        std::cout << "test_hyperplane: all tests passed\n";
        return 0;
    }
    std::cerr << "test_hyperplane: " << g_failures << " failure(s)\n";
    return 1;
}