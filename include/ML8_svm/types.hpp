// =============================================================================
// types.hpp  --  Core type aliases for the ML8_svm library
// Owner: Trevor (M6)
//
// WHY THIS FILE EXISTS
//   Every module (dataset, scaler, hyperplane, optimizer, ...) passes numbers
//   around as vectors and matrices. If each member invented their own type, the
//   modules would not fit together. So we define the two shared types ONCE here.
//
// This file is header-only (no .cpp), because it only contains type aliases.
// =============================================================================
#pragma once   // Make sure the compiler reads this file only once per build.

#include <vector>

namespace ml8_svm {

// A Vector is a list of real numbers.
// Used for: one sample's features (x), the weight vector (w), gradients, etc.
// Example: the point (1.5, -2.0) is  Vector{1.5, -2.0}
using Vector = std::vector<double>;

// A Matrix is a list of Vectors, stored ROW-MAJOR:
//   X[i]    is the i-th row = the features of sample i
//   X[i][j] is feature j of sample i
// So X.size() is the number of samples (n) and X[0].size() is the number of
// features (d). Labels (y) are NOT stored in the Matrix; they live in a
// separate std::vector<int> holding -1 or +1 for each sample.
using Matrix = std::vector<Vector>;

}  // namespace ml8_svm
