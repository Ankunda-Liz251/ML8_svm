// metrics.cpp - Model evaluation for binary classification
#include <ML8_svm/metrics.hpp>
#include <ML8_svm/errors.hpp>

#include <iomanip>
#include <sstream>
#include <string>

namespace ml8_svm {

namespace {

// Throws if the two label vectors cannot be compared.
void validate_labels(const std::vector<int>& y_true,
                     const std::vector<int>& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw DimensionMismatch(
            "metrics: y_true has " + std::to_string(y_true.size()) +
            " labels but y_pred has " + std::to_string(y_pred.size()));
    }
    if (y_true.empty()) {
        throw InvalidArgument("metrics: label vectors are empty");
    }
    for (std::size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i] != -1 && y_true[i] != 1) {
            throw InvalidArgument("metrics: y_true[" + std::to_string(i) +
                                  "] = " + std::to_string(y_true[i]) +
                                  " is not -1 or +1");
        }
        if (y_pred[i] != -1 && y_pred[i] != 1) {
            throw InvalidArgument("metrics: y_pred[" + std::to_string(i) +
                                  "] = " + std::to_string(y_pred[i]) +
                                  " is not -1 or +1");
        }
    }
}

// num / den, or 0.0 when den == 0 (zero-division convention).
double safe_ratio(std::size_t num, std::size_t den) {
    return den == 0 ? 0.0
                    : static_cast<double>(num) / static_cast<double>(den);
}

}  // namespace

ConfusionMatrix confusion_matrix(const std::vector<int>& y_true,
                                 const std::vector<int>& y_pred) {
    validate_labels(y_true, y_pred);

    ConfusionMatrix cm;
    for (std::size_t i = 0; i < y_true.size(); ++i) {
        const bool actual_pos = (y_true[i] == 1);
        const bool pred_pos = (y_pred[i] == 1);
        if (actual_pos && pred_pos)        ++cm.tp;
        else if (!actual_pos && pred_pos)  ++cm.fp;
        else if (!actual_pos && !pred_pos) ++cm.tn;
        else                               ++cm.fn;
    }
    return cm;
}

double accuracy(const std::vector<int>& y_true,
                const std::vector<int>& y_pred) {
    const ConfusionMatrix cm = confusion_matrix(y_true, y_pred);
    return safe_ratio(cm.tp + cm.tn, cm.tp + cm.fp + cm.tn + cm.fn);
}

double precision(const std::vector<int>& y_true,
                 const std::vector<int>& y_pred) {
    const ConfusionMatrix cm = confusion_matrix(y_true, y_pred);
    return safe_ratio(cm.tp, cm.tp + cm.fp);
}

double recall(const std::vector<int>& y_true,
              const std::vector<int>& y_pred) {
    const ConfusionMatrix cm = confusion_matrix(y_true, y_pred);
    return safe_ratio(cm.tp, cm.tp + cm.fn);
}

double f1_score(const std::vector<int>& y_true,
                const std::vector<int>& y_pred) {
    const double p = precision(y_true, y_pred);
    const double r = recall(y_true, y_pred);
    if (p + r == 0.0) return 0.0;
    return 2.0 * p * r / (p + r);
}

std::string classification_report(const std::vector<int>& y_true,
                                  const std::vector<int>& y_pred) {
    const ConfusionMatrix cm = confusion_matrix(y_true, y_pred);
    const std::size_t total = cm.tp + cm.fp + cm.tn + cm.fn;

    std::ostringstream out;
    out << "Classification report (positive class = +1)\n"
        << "-------------------------------------------\n"
        << "Samples: " << total << "\n\n"
        << "Confusion matrix:\n"
        << std::setw(14) << "" << std::setw(10) << "pred +1"
        << std::setw(10) << "pred -1" << "\n"
        << std::setw(14) << "actual +1" << std::setw(10) << cm.tp
        << std::setw(10) << cm.fn << "\n"
        << std::setw(14) << "actual -1" << std::setw(10) << cm.fp
        << std::setw(10) << cm.tn << "\n\n"
        << std::fixed << std::setprecision(4)
        << "Accuracy : " << accuracy(y_true, y_pred) << "\n"
        << "Precision: " << precision(y_true, y_pred) << "\n"
        << "Recall   : " << recall(y_true, y_pred) << "\n"
        << "F1 score : " << f1_score(y_true, y_pred) << "\n";
    return out.str();
}

}  // namespace ml8_svm