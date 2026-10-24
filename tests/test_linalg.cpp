// Unit tests for linalg.hpp: normal cases, empty vectors and mismatched sizes.
#include <string>

#include <ML8_svm/linalg.hpp>

#include "test_utils.hpp"

using namespace std;
using namespace ml8_svm;

int main() {
    // ---- dot ----
    ASSERT_NEAR(dot({1, 2, 3}, {4, 5, 6}), 32.0, 1e-12);
    ASSERT_NEAR(dot({1, -2}, {-3, 4}), -11.0, 1e-12);  // negative entries
    ASSERT_NEAR(dot({}, {}), 0.0, 1e-12);              // two empty vectors
    EXPECT_THROWS(dot({1, 2}, {1}), DimensionMismatch);
    EXPECT_THROWS(dot({}, {1}), DimensionMismatch);    // one empty vector

    // ---- norm ----
    ASSERT_NEAR(norm({3, 4}), 5.0, 1e-12);
    ASSERT_NEAR(norm({-3, -4}), 5.0, 1e-12);           // sign does not matter
    ASSERT_NEAR(norm({5}), 5.0, 1e-12);
    ASSERT_NEAR(norm({0, 0, 0}), 0.0, 1e-12);
    ASSERT_NEAR(norm({}), 0.0, 1e-12);

    // ---- add and sub ----
    ASSERT_TRUE(add({1, 2}, {3, 4}) == Vector({4, 6}));
    ASSERT_TRUE(add({}, {}).empty());
    ASSERT_TRUE(sub({5, 7}, {2, 3}) == Vector({3, 4}));
    ASSERT_TRUE(sub({2, 3}, {2, 3}) == Vector({0, 0}));
    ASSERT_TRUE(sub({}, {}).empty());
    EXPECT_THROWS(add({1}, {1, 2}), DimensionMismatch);
    EXPECT_THROWS(add({}, {1}), DimensionMismatch);
    EXPECT_THROWS(sub({1, 2}, {1}), DimensionMismatch);
    EXPECT_THROWS(sub({1}, {}), DimensionMismatch);

    // ---- scale ----
    ASSERT_TRUE(scale({1, -2, 3}, 2.0) == Vector({2, -4, 6}));
    ASSERT_TRUE(scale({1, -2}, -1.0) == Vector({-1, 2}));
    ASSERT_TRUE(scale({1, 2}, 0.0) == Vector({0, 0}));
    ASSERT_TRUE(scale({}, 5.0).empty());

    // ---- check_same_size ----
    check_same_size({1, 2}, {3, 4}, "test");
    check_same_size({}, {}, "test");
    EXPECT_THROWS(check_same_size({1, 2}, {3}, "test"), DimensionMismatch);

    // ---- check_X_y ----
    check_X_y({{1, 2}, {3, 4}}, {1, -1}, "test");
    EXPECT_THROWS(check_X_y({}, {}, "test"), InvalidArgument);          // no samples
    EXPECT_THROWS(check_X_y({}, {1}, "test"), InvalidArgument);         // X empty, y not
    EXPECT_THROWS(check_X_y({{1, 2}}, {}, "test"), DimensionMismatch);  // y empty, X not
    EXPECT_THROWS(check_X_y({{1, 2}, {3, 4}}, {1}, "test"), DimensionMismatch);
    EXPECT_THROWS(check_X_y({{1, 2}, {3}}, {1, -1}, "test"), DimensionMismatch);
    EXPECT_THROWS(check_X_y({{1, 2}, {3, 4}, {5}}, {1, -1, 1}, "test"),
                  DimensionMismatch);  // only the last row is ragged

    // ---- every error can be caught as the base class SvmError ----
    EXPECT_THROWS(dot({1}, {1, 2}), SvmError);
    EXPECT_THROWS(check_X_y({}, {}, "test"), SvmError);

    // ---- the error message names the function that failed ----
    try {
        dot({1, 2}, {1});
        ASSERT_TRUE(false);  // dot must throw before this line
    } catch (const DimensionMismatch& e) {
        ASSERT_TRUE(string(e.what()).find("dot") != string::npos);
    }

    return 0;
}
