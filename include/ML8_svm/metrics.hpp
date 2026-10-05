#pragma once
// metrics.hpp - Model evaluation for binary classification (M10)
//
// Labels are encoded as -1 / +1. The POSITIVE class is +1.
// All functions live in namespace ml8_svm.
//
// Error behaviour (implemented in Week 2):
//   - y_true.size() != y_pred.size()  -> throws DimensionMismatch
//   - empty input                     -> throws InvalidArgument
//   - any label not in {-1, +1}       -> throws InvalidArgument
//
// Zero-division convention (implemented in Week 2):
//   precision = 0.0 when tp + fp == 0
//   recall    = 0.0 when tp + fn == 0
//   f1_score  = 0.0 when precision + recall == 0

#include <cstddef>
#include <string>
#include <vector>

namespace ml8_svm {

/// Counts of the four outcomes, with +1 as the positive class.
struct ConfusionMatrix {
    std::size_t tp{0};  // true  = +1, predicted = +1
    std::size_t fp{0};  // true  = -1, predicted = +1
    std::size_t tn{0};  // true  = -1, predicted = -1
    std::size_t fn{0};  // true  = +1, predicted = -1
};

/// Build the confusion matrix from true and predicted labels.
ConfusionMatrix confusion_matrix(const std::vector<int>& y_true,
                                 const std::vector<int>& y_pred);

/// (tp + tn) / total
double accuracy(const std::vector<int>& y_true,
                const std::vector<int>& y_pred);

/// tp / (tp + fp)
double precision(const std::vector<int>& y_true,
                 const std::vector<int>& y_pred);

/// tp / (tp + fn)
double recall(const std::vector<int>& y_true,
              const std::vector<int>& y_pred);

/// 2 * precision * recall / (precision + recall)
double f1_score(const std::vector<int>& y_true,
                const std::vector<int>& y_pred);

/// Human-readable text report: confusion matrix + accuracy, precision,
/// recall and F1.
std::string classification_report(const std::vector<int>& y_true,
                                  const std::vector<int>& y_pred);

}  // namespace ml8_svm