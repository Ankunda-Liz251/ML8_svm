

//  test_linear_svm.cpp  
//  Tests for the LinearSVM class: errors, training, predicting.
//
//  Uses test_assert() from test_utils.hpp (M3). A failed test_assert throws,
//  so the program stops with a non-zero exit code and ctest reports a failure.

#include <ML8_svm/linear_svm.hpp>
#include <ML8_svm/errors.hpp>
#include "test_utils.hpp"

#include <cmath>

using namespace ml8_svm;

static bool almost_equal(double a, double b, double eps = 1e-12) {
    return std::fabs(a - b) <= eps;
}

// Passes ONLY if the code throws exactly the expected exception type.
// (The shared expect_throw in test_utils.hpp also passes when nothing is
//  thrown, so this local helper is safer. See the note in the Week 2 report.)
template <typename ExceptionType, typename Function>
static void expect_throw_type(Function function, const std::string& message) {
    bool right_type = false;
    try { function(); }
    catch (const ExceptionType&) { right_type = true; }
    catch (...) {}
    test_assert(right_type, message);
}

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

    //1. constructor rejects bad settings
    expect_throw_type<InvalidArgument>(
        [&] { LinearSVM bad(-1.0, good_config()); },
        "negative lambda throws InvalidArgument");
    expect_throw_type<InvalidArgument>(
        [&] { LinearSVM bad(std::nan(""), good_config()); },
        "NaN lambda throws InvalidArgument");
    expect_throw_type<InvalidArgument>(
        [&] { OptimizerConfig c = good_config(); c.learning_rate = 0.0; LinearSVM bad(lambda, c); },
        "learning rate 0 throws InvalidArgument");
    expect_throw_type<InvalidArgument>(
        [&] { OptimizerConfig c = good_config(); c.epochs = 0; LinearSVM bad(lambda, c); },
        "zero epochs throws InvalidArgument");

    //  2. everything throws NotFitted before fit() 
    {
        LinearSVM m(lambda, good_config());
        expect_throw_type<NotFitted>([&] { m.predict(Vector{1, 1}); }, "predict(one) before fit throws NotFitted");
        expect_throw_type<NotFitted>([&] { m.predict(toy_X()); }, "predict(matrix) before fit throws NotFitted");
        expect_throw_type<NotFitted>([&] { m.decision_function(Vector{1, 1}); }, "decision_function(one) before fit throws NotFitted");
        expect_throw_type<NotFitted>([&] { m.decision_function(toy_X()); }, "decision_function(matrix) before fit throws NotFitted");
        expect_throw_type<NotFitted>([&] { m.hyperplane(); }, "hyperplane() before fit throws NotFitted");
        expect_throw_type<NotFitted>([&] { m.history(); }, "history() before fit throws NotFitted");
        expect_throw_type<NotFitted>([&] { m.support_vector_indices(); }, "support_vector_indices() before fit throws NotFitted");
    }

    // 3. fit() rejects bad training data 
    {
        LinearSVM m(lambda, good_config());
        expect_throw_type<InvalidArgument>([&] { m.fit(Matrix{}, std::vector<int>{}); }, "empty data throws InvalidArgument");
        expect_throw_type<DimensionMismatch>([&] { m.fit(toy_X(), std::vector<int>{1, -1}); }, "label count mismatch throws DimensionMismatch");
        expect_throw_type<DimensionMismatch>([&] { m.fit(Matrix{{1, 2}, {3}}, std::vector<int>{1, -1}); }, "ragged rows throw DimensionMismatch");
        expect_throw_type<InvalidArgument>([&] { m.fit(Matrix{{1, 2}, {3, 4}}, std::vector<int>{1, 0}); }, "label 0 throws InvalidArgument");
        expect_throw_type<InvalidArgument>([&] { m.fit(Matrix{{1, 2}, {3, 4}}, std::vector<int>{1, 1}); }, "one class only throws InvalidArgument");
        expect_throw_type<InvalidArgument>([&] { m.fit(Matrix{{1, std::nan("")}, {3, 4}}, std::vector<int>{1, -1}); }, "NaN feature throws InvalidArgument");
        expect_throw_type<InvalidArgument>([&] { m.fit(Matrix{{}, {}}, std::vector<int>{1, -1}); }, "zero features throws InvalidArgument");
        // none of the failed calls may leave the model "fitted"
        expect_throw_type<NotFitted>([&] { m.predict(Vector{1, 1}); }, "failed fits leave the model not fitted");
    }

    // 4. a good fit solves the toy probem 
    LinearSVM model(lambda, good_config());
    model.fit(toy_X(), toy_y());

    std::vector<int> predicted = model.predict(toy_X());
    test_assert(predicted == toy_y(), "toy dataset is classified with 100% accuracy");
    test_assert(model.predict(Vector{4.0, 4.0}) == +1, "unseen point (4,4) is +1");
    test_assert(model.predict(Vector{-4.0, -4.0}) == -1, "unseen point (-4,-4) is -1");

    // predict must agree with the sign of decision_function
    Vector scores = model.decision_function(toy_X());
    bool signs_agree = true, versions_agree = true;
    for (size_t i = 0; i < scores.size(); ++i) {
        if ((scores[i] >= 0) != (predicted[i] == +1)) signs_agree = false;
        if (!almost_equal(scores[i], model.decision_function(toy_X()[i]))) versions_agree = false;
    }
    test_assert(signs_agree, "predict agrees with the sign of decision_function");
    test_assert(versions_agree, "matrix and single-sample scores agree");

    // 5. accessors after training 
    test_assert(model.hyperplane().w.size() == 2, "hyperplane has one weight per feature");
    test_assert(model.history().epochs_run >= 1, "history records at least one epoch");
    test_assert(!model.history().loss.empty(), "history records the loss");
    test_assert(model.history().loss.back() <= model.history().loss.front(), "loss went down during training");
    bool indices_ok = true;
    for (size_t idx : model.support_vector_indices()) if (idx >= toy_X().size()) indices_ok = false;
    test_assert(indices_ok, "support vector indices are valid row numbers");

    // 6. wrong feature count at prediction time 
    expect_throw_type<DimensionMismatch>([&] { model.predict(Vector{1.0}); }, "predict with wrong feature count throws DimensionMismatch");
    expect_throw_type<DimensionMismatch>([&] { model.predict(Matrix{{1.0, 2.0}, {1.0}}); }, "predict matrix with a short row throws DimensionMismatch");

    // 7. a failed re-fit keeps the old working model 
    Hyperplane before = model.hyperplane();
    expect_throw_type<InvalidArgument>([&] { model.fit(Matrix{}, std::vector<int>{}); }, "bad re-fit throws InvalidArgument");
    test_assert(model.hyperplane().w == before.w && model.hyperplane().b == before.b, "failed re-fit leaves the line unchanged");
    test_assert(model.predict(toy_X()) == toy_y(), "model still predicts correctly after a failed re-fit");

    //  8. fitting again on new data replaces the model
    std::vector<int> flipped = toy_y();
    for (int& label : flipped) label = -label;
    model.fit(toy_X(), flipped);
    test_assert(model.predict(toy_X()) == flipped, "second fit replaces the old model");

    return 0;   // reached only if every test_assert passed
}