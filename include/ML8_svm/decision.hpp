
//  decision.hpp          

//  WHAT THIS FILE IS FOR
//  A trained SVM is just a straight line (in 2-D) / flat plane (in higher
//  dimensions) described by a weight vector `w` and a bias number `b`.
//
//  For any data point x we compute a single number called the SCORE:
//
//        score(x) = w . x + b
//
//    * score > 0  -> the point is on the "+1" side of the line
//    * score < 0  -> the point is on the "-1" side of the line
//    * score = 0  -> the point is exactly on the line
//    * the bigger |score| is, the farther the point is from the line
//      (so the model is more "sure" about it)
//
//  The functions below turn a hyperplane + data into scores, and scores into
//  class labels (+1 or -1). A prediction is simply "which side are you on?".
//
//  EXAMPLE
//      Hyperplane h;  h.w = {2.0, -1.0};  h.b = 0.5;
//      double s = decision_function(h, Vector{1.0, 1.0});   // 2 - 1 + 0.5 = 1.5
//      int label = to_label(s);                             // +1
//
//  COST
//  One sample costs O(d) time, where d is the number of features. A matrix of
//  n samples costs O(n * d). No extra memory is used besides the result.
//
//  Most users do not call these functions directly: LinearSVM (linear_svm.hpp)
//  calls them for you inside predict() and decision_function().

#pragma once

#include <ML8_svm/hyperplane.hpp>   // the Hyperplane struct (w and b) - written by M4
#include <ML8_svm/types.hpp>        // Vector and Matrix - written by M6

namespace ml8_svm {

// Score of ONE sample x.
//   hyperplane : the trained line (w, b)
//   x          : one sample, e.g. {1.5, 2.0}
//   returns    : w . x + b
// Throws DimensionMismatch if x does not have the same number of features as w.
// Throws nothing else; the work is done by Hyperplane::score (no duplicated maths).
// Note: if x contains NaN the result is NaN. It is to_label() that rejects NaN.
double decision_function(const Hyperplane& hyperplane, const Vector& x);

// Scores of MANY samples (a Matrix has one row per sample).
//   returns : one score per row, in the same order.
// An empty matrix gives an empty result.
// Throws DimensionMismatch if any row has the wrong number of features.
Vector decision_function(const Hyperplane& hyperplane, const Matrix& X);

// Turn a score into a class label.
//   score >= 0  -> +1
//   score <  0  -> -1
// A score of exactly 0 is "on the line"; we must pick a side, so we choose +1
// (this is the rule fixed in the project's maths section).
// Infinity is allowed: +infinity gives +1 and -infinity gives -1.
// Throws InvalidArgument if score is NaN (not a number), because a NaN would
// otherwise silently become -1 and hide a bug.
int to_label(double score);

}  // namespace ml8_svm