// =============================================================================
//  test_linear_svm.cpp   Owner: Liz (M9)
//  Tests for the LinearSVM class: errors, training, predicting.
// =============================================================================
#include <ML8_svm/linear_svm.hpp>
#include <ML8_svm/errors.hpp>
#include "test_utils.hpp"

#include <cmath>

using namespace ml8_svm;

// Four points top-right (+1), four bottom-left (-1): a line separates them.
static Matrix toy_X() {
    return {{ 2,  2}, { 3,  3}, { 2,  3}, { 3,  2},
            {-2, -2}, {-3, -3}, {-2, -3}, {-3, -2}};
}
static std::vector<int> toy_y() { return {+1, +1, +1, +1, -1, -1, -1, -1}; }

static OptimizerConfig good_config() {
    // learning_rate, epochs, batch_size, tol, decay, seed
    return OptimizerConfig{0.1, 500, 8, 1e-9, false, 42};
}

int main() {
    const double lambda = 0.01;

    // --- 1. constructor rejects bad settings ---------------------------------
    TEST_EXPECT_THROWS(LinearSVM(-1.0, good_config()), InvalidArgument);
    TEST_EXPECT_THROWS(LinearSVM(std::nan(""), good_config()), InvalidArgument);
    { OptimizerConfig c = good_config(); c.learning_rate = 0.0;
      TEST_EXPECT_THROWS(LinearSVM(lambda, c), InvalidArgument); }
    { OptimizerConfig c = good_config(); c.epochs = 0;
      TEST_EXPECT_THROWS(LinearSVM(lambda, c), InvalidArgument); }

    // --- 2. everything throws NotFitted before fit() -------------------------
    {
        LinearSVM m(lambda, good_config());
        TEST_EXPECT_THROWS(m.predict(Vector{1, 1}), NotFitted);
        TEST_EXPECT_THROWS(m.predict(toy_X()), NotFitted);
        TEST_EXPECT_THROWS(m.decision_function(Vector{1, 1}), NotFitted);
        TEST_EXPECT_THROWS(m.decision_function(toy_X()), NotFitted);
        TEST_EXPECT_THROWS(m.hyperplane(), NotFitted);
        TEST_EXPECT_THROWS(m.history(), NotFitted);
        TEST_EXPECT_THROWS(m.support_vector_indices(), NotFitted);
    }

    // --- 3. fit() rejects bad training data ----------------------------------
    {
        LinearSVM m(lambda, good_config());
        TEST_EXPECT_THROWS(m.fit(Matrix{}, std::vector<int>{}), InvalidArgument);
        TEST_EXPECT_THROWS(m.fit(toy_X(), std::vector<int>{1, -1}), DimensionMismatch);
        TEST_EXPECT_THROWS(m.fit(Matrix{{1, 2}, {3}}, std::vector<int>{1, -1}), DimensionMismatch);
        TEST_EXPECT_THROWS(m.fit(Matrix{{1, 2}, {3, 4}}, std::vector<int>{1, 0}), InvalidArgument);
        TEST_EXPECT_THROWS(m.fit(Matrix{{1, 2}, {3, 4}}, std::vector<int>{1, 1}), InvalidArgument);
        TEST_EXPECT_THROWS(m.fit(Matrix{{1, std::nan("")}, {3, 4}}, std::vector<int>{1, -1}), InvalidArgument);
        TEST_EXPECT_THROWS(m.fit(Matrix{{}, {}}, std::vector<int>{1, -1}), InvalidArgument);
        // none of the failed calls may leave the model "fitted"
        TEST_EXPECT_THROWS(m.predict(Vector{1, 1}), NotFitted);
    }

    // --- 4. a good fit solves the toy problem 100% ---------------------------
    LinearSVM model(lambda, good_config());
    model.fit(toy_X(), toy_y());

    std::vector<int> predicted = model.predict(toy_X());
    TEST_ASSERT(predicted == toy_y());                       // 100% accuracy
    TEST_ASSERT(model.predict(Vector{4.0, 4.0}) == +1);      // unseen points
    TEST_ASSERT(model.predict(Vector{-4.0, -4.0}) == -1);

    // predict must agree with the sign of decision_function
    Vector scores = model.decision_function(toy_X());
    for (size_t i = 0; i < scores.size(); ++i) {
        TEST_ASSERT((scores[i] >= 0) == (predicted[i] == +1));
        TEST_ASSERT_NEAR(scores[i], model.decision_function(toy_X()[i]), 1e-12);
    }

    // --- 5. accessors after training ------------------------------------------
    TEST_ASSERT(model.hyperplane().w.size() == 2);
    TEST_ASSERT(model.history().epochs_run >= 1);
    TEST_ASSERT(!model.history().loss.empty());
    TEST_ASSERT(model.history().loss.back() <= model.history().loss.front());
    for (size_t idx : model.support_vector_indices()) TEST_ASSERT(idx < toy_X().size());

    // --- 6. wrong feature count at prediction time -----------------------------
    TEST_EXPECT_THROWS(model.predict(Vector{1.0}), DimensionMismatch);
    TEST_EXPECT_THROWS(model.predict(Matrix{{1.0, 2.0}, {1.0}}), DimensionMismatch);

    // --- 7. a failed re-fit keeps the old working model --------------------------
    Hyperplane before = model.hyperplane();
    TEST_EXPECT_THROWS(model.fit(Matrix{}, std::vector<int>{}), InvalidArgument);
    TEST_ASSERT(model.hyperplane().w == before.w && model.hyperplane().b == before.b);
    TEST_ASSERT(model.predict(toy_X()) == toy_y());

    // --- 8. fitting again on new data replaces the model -------------------------
    std::vector<int> flipped = toy_y();
    for (int& label : flipped) label = -label;
    model.fit(toy_X(), flipped);
    TEST_ASSERT(model.predict(toy_X()) == flipped);

    TEST_FINISH();
}