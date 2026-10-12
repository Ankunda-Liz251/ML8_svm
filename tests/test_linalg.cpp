// Unit tests for linalg.hpp.
#include <ML8_svm/linalg.hpp>

#include "test_utils.hpp"

using namespace ml8_svm;

int main() {
    // dot
    ASSERT_NEAR(dot({1, 2, 3}, {4, 5, 6}), 32.0, 1e-12);
    ASSERT_NEAR(dot({}, {}), 0.0, 1e-12);
    EXPECT_THROWS(dot({1, 2}, {1}), DimensionMismatch);

    // norm
    ASSERT_NEAR(norm({3, 4}), 5.0, 1e-12);
    ASSERT_NEAR(norm({0, 0, 0}), 0.0, 1e-12);
    ASSERT_NEAR(norm({}), 0.0, 1e-12);

    // add and sub
    ASSERT_TRUE(add({1, 2}, {3, 4}) == Vector({4, 6}));
    ASSERT_TRUE(sub({5, 7}, {2, 3}) == Vector({3, 4}));
    EXPECT_THROWS(add({1}, {1, 2}), DimensionMismatch);
    EXPECT_THROWS(sub({1, 2}, {1}), DimensionMismatch);

    // scale
    ASSERT_TRUE(scale({1, -2, 3}, 2.0) == Vector({2, -4, 6}));
    ASSERT_TRUE(scale({1, 2}, 0.0) == Vector({0, 0}));
    ASSERT_TRUE(scale({}, 5.0).empty());

    // check_same_size
    check_same_size({1, 2}, {3, 4}, "test");
    EXPECT_THROWS(check_same_size({1, 2}, {3}, "test"), DimensionMismatch);

    // check_X_y
    check_X_y({{1, 2}, {3, 4}}, {1, -1}, "test");
    EXPECT_THROWS(check_X_y({}, {}, "test"), InvalidArgument);
    EXPECT_THROWS(check_X_y({{1, 2}, {3, 4}}, {1}, "test"), DimensionMismatch);
    EXPECT_THROWS(check_X_y({{1, 2}, {3}}, {1, -1}, "test"), DimensionMismatch);

    return 0;
}
