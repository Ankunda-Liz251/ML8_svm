// test_metrics.cpp - Unit tests for metrics.hpp (M10)
#include <ML8_svm/metrics.hpp>
#include <ML8_svm/errors.hpp>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace ml8_svm;

//Minimal local helpers (swap for tests/test_utils.hpp when available)
static int g_failures = 0;

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::cerr << "FAILED: " #cond " (" << __FILE__ << ":"        \
                      << __LINE__ << ")\n";                               \
            ++g_failures;                                                 \
        }                                                                 \
    } while (0)

#define CHECK_NEAR(a, b) CHECK(std::fabs((a) - (b)) < 1e-12)

#define EXPECT_THROWS(ExType, ...)                                        \
    do {                                                                  \
        bool caught = false;                                              \
        try { (void)(__VA_ARGS__); }                                      \
        catch (const ExType&) { caught = true; }                          \
        catch (...) {}                                                    \
        if (!caught) {                                                    \
            std::cerr << "FAILED: expected " #ExType " from " #__VA_ARGS__ \
                      << " (" << __FILE__ << ":" << __LINE__ << ")\n";    \
            ++g_failures;                                                 \
        }                                                                 \
    } while (0)

using Labels = std::vector<int>;

// ---- Struct ----------------------------------------------------------------
static void test_confusion_matrix_struct() {
    ConfusionMatrix cm;
    CHECK(cm.tp == 0 && cm.fp == 0 && cm.tn == 0 && cm.fn == 0);
}

// ---- Confusion matrix -------------------------------------------------------
// y_true = {+1,+1,+1,-1,-1,-1,-1,+1}
// y_pred = {+1,+1,-1,-1,-1,+1,-1,+1}
//   tp: idx 0,1,7 = 3    fn: idx 2 = 1
//   fp: idx 5     = 1    tn: idx 3,4,6 = 3
static const Labels kTrue = {1, 1, 1, -1, -1, -1, -1, 1};
static const Labels kPred = {1, 1, -1, -1, -1, 1, -1, 1};

static void test_confusion_matrix_counts() {
    const ConfusionMatrix cm = confusion_matrix(kTrue, kPred);
    CHECK(cm.tp == 3);
    CHECK(cm.fn == 1);
    CHECK(cm.fp == 1);
    CHECK(cm.tn == 3);
}

// ---- Main metrics -----------------------------------------------------------
// accuracy = 6/8 = 0.75, precision = 3/4, recall = 3/4, f1 = 0.75
static void test_accuracy_precision_recall_f1() {
    CHECK_NEAR(accuracy(kTrue, kPred), 0.75);
    CHECK_NEAR(precision(kTrue, kPred), 0.75);
    CHECK_NEAR(recall(kTrue, kPred), 0.75);
    CHECK_NEAR(f1_score(kTrue, kPred), 0.75);
}

// precision and recall differ here, which catches swapped formulas.
// true = {+1,+1,+1,+1,-1,-1}, pred = {+1,+1,-1,-1,-1,-1}
//   tp=2 fn=2 fp=0 tn=2
//   precision = 2/2 = 1.0, recall = 2/4 = 0.5, f1 = 2*1*0.5/1.5 = 2/3
static void test_precision_differs_from_recall_high_precision() {
    const Labels t = {1, 1, 1, 1, -1, -1};
    const Labels p = {1, 1, -1, -1, -1, -1};
    CHECK_NEAR(precision(t, p), 1.0);
    CHECK_NEAR(recall(t, p), 0.5);
    CHECK_NEAR(f1_score(t, p), 2.0 / 3.0);
    CHECK_NEAR(accuracy(t, p), 4.0 / 6.0);
}

// true = {+1,-1,-1,-1}, pred = {+1,+1,+1,-1}
//   tp=1 fp=2 tn=1 fn=0
//   precision = 1/3, recall = 1/1 = 1.0, f1 = 2*(1/3)*1/(4/3) = 0.5
static void test_precision_differs_from_recall_high_recall() {
    const Labels t = {1, -1, -1, -1};
    const Labels p = {1, 1, 1, -1};
    CHECK_NEAR(precision(t, p), 1.0 / 3.0);
    CHECK_NEAR(recall(t, p), 1.0);
    CHECK_NEAR(f1_score(t, p), 0.5);
}

static void test_perfect_predictions() {
    const Labels t = {1, -1, 1, -1, 1};
    CHECK_NEAR(accuracy(t, t), 1.0);
    CHECK_NEAR(precision(t, t), 1.0);
    CHECK_NEAR(recall(t, t), 1.0);
    CHECK_NEAR(f1_score(t, t), 1.0);
}

// true = {+1,+1,-1,-1}, pred = {-1,-1,+1,+1}: tp=0 fn=2 fp=2 tn=0
static void test_all_wrong_predictions() {
    const Labels t = {1, 1, -1, -1};
    const Labels p = {-1, -1, 1, 1};
    CHECK_NEAR(accuracy(t, p), 0.0);
    CHECK_NEAR(precision(t, p), 0.0);
    CHECK_NEAR(recall(t, p), 0.0);
    CHECK_NEAR(f1_score(t, p), 0.0);
}

// ---- Zero-division cases ----------------------------------------------------
// Nothing predicted positive: tp + fp == 0 -> precision = 0.0
// true = {+1,+1,-1,-1}, pred = all -1: tp=0 fp=0 tn=2 fn=2
static void test_zero_division_precision() {
    const Labels t = {1, 1, -1, -1};
    const Labels p = {-1, -1, -1, -1};
    CHECK_NEAR(precision(t, p), 0.0);
    CHECK_NEAR(recall(t, p), 0.0);       // 0 / (0 + 2)
    CHECK_NEAR(accuracy(t, p), 0.5);
    CHECK_NEAR(f1_score(t, p), 0.0);
}

// No actual positives: tp + fn == 0 -> recall = 0.0
// true = all -1, pred = {+1,-1,-1}: tp=0 fp=1 tn=2 fn=0
static void test_zero_division_recall() {
    const Labels t = {-1, -1, -1};
    const Labels p = {1, -1, -1};
    CHECK_NEAR(recall(t, p), 0.0);
    CHECK_NEAR(precision(t, p), 0.0);    // 0 / (0 + 1)
    CHECK_NEAR(accuracy(t, p), 2.0 / 3.0);
    CHECK_NEAR(f1_score(t, p), 0.0);
}

// Only negatives, all predicted negative: precision, recall and F1 are all
// zero-division cases, accuracy is perfect.
static void test_only_negatives_all_correct() {
    const Labels t = {-1, -1, -1};
    CHECK_NEAR(accuracy(t, t), 1.0);
    CHECK_NEAR(precision(t, t), 0.0);
    CHECK_NEAR(recall(t, t), 0.0);
    CHECK_NEAR(f1_score(t, t), 0.0);
}

// ---- Invalid input ----------------------------------------------------------
static void test_size_mismatch_throws() {
    const Labels t = {1, -1, 1};
    const Labels p = {1, -1};
    EXPECT_THROWS(DimensionMismatch, confusion_matrix(t, p));
    EXPECT_THROWS(DimensionMismatch, accuracy(t, p));
    EXPECT_THROWS(DimensionMismatch, precision(t, p));
    EXPECT_THROWS(DimensionMismatch, recall(t, p));
    EXPECT_THROWS(DimensionMismatch, f1_score(t, p));
    EXPECT_THROWS(DimensionMismatch, classification_report(t, p));
}

static void test_empty_input_throws() {
    const Labels e;
    EXPECT_THROWS(InvalidArgument, confusion_matrix(e, e));
    EXPECT_THROWS(InvalidArgument, accuracy(e, e));
    EXPECT_THROWS(InvalidArgument, f1_score(e, e));
}

static void test_invalid_label_throws() {
    const Labels good = {1, -1, 1};
    const Labels zero = {1, 0, 1};
    const Labels two = {1, -1, 2};
    EXPECT_THROWS(InvalidArgument, confusion_matrix(zero, good));  // bad y_true
    EXPECT_THROWS(InvalidArgument, confusion_matrix(good, two));   // bad y_pred
    EXPECT_THROWS(InvalidArgument, accuracy(zero, two));
}

// ---- Report -----------------------------------------------------------------
static void test_classification_report_contains_metrics() {
    const std::string r = classification_report(kTrue, kPred);
    CHECK(r.find("Accuracy") != std::string::npos);
    CHECK(r.find("Precision") != std::string::npos);
    CHECK(r.find("Recall") != std::string::npos);
    CHECK(r.find("F1") != std::string::npos);
    CHECK(r.find("0.7500") != std::string::npos);
}

int main() {
    test_confusion_matrix_struct();
    test_confusion_matrix_counts();
    test_accuracy_precision_recall_f1();
    test_precision_differs_from_recall_high_precision();
    test_precision_differs_from_recall_high_recall();
    test_perfect_predictions();
    test_all_wrong_predictions();
    test_zero_division_precision();
    test_zero_division_recall();
    test_only_negatives_all_correct();
    test_size_mismatch_throws();
    test_empty_input_throws();
    test_invalid_label_throws();
    test_classification_report_contains_metrics();

    if (g_failures == 0) {
        std::cout << "test_metrics: all checks passed\n";
        return 0;
    }
    std::cerr << "test_metrics: " << g_failures << " check(s) FAILED\n";
    return 1;
}