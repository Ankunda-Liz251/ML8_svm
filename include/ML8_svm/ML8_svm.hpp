#ifndef ML8_SVM_ML8_SVM_HPP
#define ML8_SVM_ML8_SVM_HPP

// Umbrella header: include this to get the whole ML8_svm library.
//     #include <ML8_svm/ML8_svm.hpp>

// Core layer
#include <ML8_svm/types.hpp>
#include <ML8_svm/errors.hpp>
#include <ML8_svm/linalg.hpp>

// Data handling
#include <ML8_svm/dataset.hpp>
#include <ML8_svm/preprocessing.hpp>
#include <ML8_svm/scaler.hpp>

// Model components
#include <ML8_svm/hyperplane.hpp>
#include <ML8_svm/hinge_loss.hpp>
#include <ML8_svm/regularization.hpp>
#include <ML8_svm/optimizer.hpp>
#include <ML8_svm/support_vectors.hpp>
#include <ML8_svm/decision.hpp>
#include <ML8_svm/linear_svm.hpp>

// Evaluation
#include <ML8_svm/metrics.hpp>

#endif  // ML8_SVM_ML8_SVM_HPP