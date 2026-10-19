
//  test_linear_svm.cpp  
//  Tests for the LinearSVM class: errors, training, predicting.
//
//  Uses test_assert() from test_utils.hpp (M3). A failed test_assert throws,
//  so the program stops with a non-zero exit code and ctest reports a failure.

#include <ML8_svm/linear_svm.hpp>
#include <ML8_svm/errors.hpp>
#include "test_utils.hpp"

#include <cmath>
#include <limits>

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

    // 2. everything throws NotFitted before fit()
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

    //  4. a good fit solves the toy problem 100% 
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

    //  5. accessors after training 
    test_assert(model.hyperplane().w.size() == 2, "hyperplane has one weight per feature");
    test_assert(model.history().epochs_run >= 1, "history records at least one epoch");
    test_assert(!model.history().loss.empty(), "history records the loss");
    test_assert(model.history().loss.back() <= model.history().loss.front(), "loss went down during training");
    bool indices_ok = true;
    for (size_t idx : model.support_vector_indices()) if (idx >= toy_X().size()) indices_ok = false;
    test_assert(indices_ok, "support vector indices are valid row numbers");

    //  6. wrong feature count at prediction time 
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

    // WEEK 3: EDGE CASES 

    //9. tol must be a sensible number 
    expect_throw_type<InvalidArgument>(
        [&] { OptimizerConfig c = good_config(); c.tol = -1.0; LinearSVM bad(lambda, c); },
        "negative tol throws InvalidArgument");
    expect_throw_type<InvalidArgument>(
        [&] { OptimizerConfig c = good_config(); c.tol = std::nan(""); LinearSVM bad(lambda, c); },
        "NaN tol throws InvalidArgument");

    //  10. predicting on an empty matrix gives an empty answer 
    test_assert(model.predict(Matrix{}).empty(), "predict on empty matrix gives empty result");
    test_assert(model.decision_function(Matrix{}).empty(), "decision_function on empty matrix gives empty result");

    //  11. NaN in a prediction input is an error, not a silent label 
    expect_throw_type<InvalidArgument>(
        [&] { model.predict(Vector{std::nan(""), 1.0}); },
        "predict with NaN feature throws InvalidArgument");

    //  12. a copy of a trained model predicts the same 
    {
        LinearSVM copy = model;
        test_assert(copy.predict(toy_X()) == model.predict(toy_X()), "a copied model predicts the same labels");
    }

    // 13. history never reports more epochs than allowed 
    test_assert(model.history().epochs_run <= good_config().epochs, "epochs_run is within the epoch limit");

    //  14. one-feature (1-D) data works 
    {
        LinearSVM one_d(lambda, good_config());
        one_d.fit(Matrix{{1.0}, {2.0}, {3.0}, {-1.0}, {-2.0}, {-3.0}},
                  std::vector<int>{+1, +1, +1, -1, -1, -1});
        test_assert(one_d.hyperplane().w.size() == 1, "1-D model has one weight");
        test_assert(one_d.predict(Vector{5.0}) == +1 && one_d.predict(Vector{-5.0}) == -1,
                    "1-D model separates positive and negative values");
    }

    // 15. refitting with a different number of features works 
    {
        LinearSVM m(lambda, good_config());
        m.fit(toy_X(), toy_y());                               // 2 features
        Matrix X3 = {{2, 2, 2}, {3, 3, 3}, {-2, -2, -2}, {-3, -3, -3}};
        m.fit(X3, std::vector<int>{+1, +1, -1, -1});           // 3 features
        test_assert(m.hyperplane().w.size() == 3, "refit changes the model to 3 weights");
        expect_throw_type<DimensionMismatch>([&] { m.predict(Vector{1.0, 1.0}); },
                                             "old 2-feature input is rejected after refit to 3 features");
    }

    //  16. the model keeps its OWN copy of the training data
    {
        Matrix X = toy_X();
        LinearSVM m(lambda, good_config());
        m.fit(X, toy_y());
        std::vector<size_t> before_indices = m.support_vector_indices();
        for (Vector& row : X) for (double& v : row) v = 1000.0;   // damage the caller's copy
        test_assert(m.support_vector_indices() == before_indices,
                    "support vectors do not change if the caller edits its own data");
    }

    // 17. training that blows up is rejected and keeps the old model
    {
        LinearSVM m(lambda, good_config());
        m.fit(toy_X(), toy_y());
        Hyperplane good = m.hyperplane();
        // learning_rate * lambda = 10 is far too big, so the numbers explode.
        LinearSVM wild(1.0, OptimizerConfig{10.0, 500, 8, 1e-9, false, 1});
        expect_throw_type<SvmError>([&] { wild.fit(toy_X(), toy_y()); },
                                    "diverging training throws instead of keeping NaN weights");
        expect_throw_type<NotFitted>([&] { wild.predict(Vector{1, 1}); },
                                     "a model whose training diverged stays not fitted");
        test_assert(m.hyperplane().w == good.w, "an unrelated good model is unaffected");
    }


    //  18. a few more edge cases 
    {
        // lambda = 0 (no regularisation) is allowed
        LinearSVM zero_lambda(0.0, good_config());
        zero_lambda.fit(toy_X(), toy_y());
        test_assert(zero_lambda.predict(toy_X()) == toy_y(), "lambda = 0 still trains and classifies the toy set");

        // labels must be exactly -1 or +1
        LinearSVM m(lambda, good_config());
        expect_throw_type<InvalidArgument>([&] { m.fit(Matrix{{1, 2}, {3, 4}}, std::vector<int>{2, -2}); },
                                           "labels 2 and -2 throw InvalidArgument");
        expect_throw_type<InvalidArgument>(
            [&] { m.fit(Matrix{{1, std::numeric_limits<double>::infinity()}, {3, 4}}, std::vector<int>{1, -1}); },
            "infinite feature throws InvalidArgument");

        // the smallest possible data set: one sample per class
        LinearSVM two_samples(lambda, good_config());
        two_samples.fit(Matrix{{1, 0}, {-1, 0}}, std::vector<int>{+1, -1});
        test_assert(two_samples.predict(Vector{2, 0}) == +1 && two_samples.predict(Vector{-2, 0}) == -1,
                    "two-sample model puts each side in the right class");

        // same data and same settings give the same line
        LinearSVM a(lambda, good_config()), b(lambda, good_config());
        a.fit(toy_X(), toy_y());
        b.fit(toy_X(), toy_y());
        test_assert(a.hyperplane().w == b.hyperplane().w && a.hyperplane().b == b.hyperplane().b,
                    "same data and same settings give the same line");

        // a copy keeps its model when the original is re-fit
        LinearSVM original(lambda, good_config());
        original.fit(toy_X(), toy_y());
        LinearSVM copy = original;
        original.fit(toy_X(), flipped);
        test_assert(copy.predict(toy_X()) == toy_y(), "a copy keeps the old model when the original is re-fit");
    }

    return 0;   // reached only if every test_assert passed
}