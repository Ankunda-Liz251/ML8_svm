// test_metrics.cpp - SKELETON (Week 1)
// Real tests are added in Week 2 once src/metrics.cpp is implemented.
#include <ML8_svm/metrics.hpp>

#include <iostream>
#include <vector>

using namespace ml8_svm;

static int g_failures = 0;
#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::cerr << "FAILED: " #cond " (" << __FILE__ << ":"         \
                      << __LINE__ << ")\n";                                \
            ++g_failures;                                                  \
        }                                                                  \
    } while (0)

// --- Runs now: header compiles, struct is usable ---------------------------
static void test_confusion_matrix_struct() {
    ConfusionMatrix cm;
    CHECK(cm.tp == 0 && cm.fp == 0 && cm.tn == 0 && cm.fn == 0);
    cm.tp = 3; cm.fn = 1;
    CHECK(cm.tp == 3 && cm.fn == 1);
}

// --- Week 2 TODO: hand-computed cases ---------------------------------------
// y_true = {+1,+1,+1,-1,-1,-1,-1,+1}
// y_pred = {+1,+1,-1,-1,-1,+1,-1,+1}
//   expected: tp=3, fn=1, fp=1, tn=3
//   accuracy = 6/8 = 0.75, precision = 3/4, recall = 3/4, f1 = 0.75
//
// TODO test_confusion_matrix_counts()
// TODO test_accuracy_precision_recall_f1()
// TODO test_perfect_predictions()          (all metrics == 1.0)
// TODO test_all_wrong_predictions()        (accuracy == 0.0)
// TODO test_zero_division_precision()      (no predicted positives -> 0.0)
// TODO test_zero_division_recall()         (no actual positives    -> 0.0)
// TODO test_f1_zero_when_p_and_r_zero()
// TODO test_size_mismatch_throws()         (DimensionMismatch)
// TODO test_empty_input_throws()           (InvalidArgument)
// TODO test_invalid_label_throws()         (label 0 or 2 -> InvalidArgument)
// TODO test_classification_report_contains_metrics()

int main() {
    test_confusion_matrix_struct();
    if (g_failures == 0) {
        std::cout << "test_metrics: all checks passed (skeleton)\n";
        return 0;
    }
    return 1;
}