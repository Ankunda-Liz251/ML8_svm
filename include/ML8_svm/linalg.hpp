// =============================================================================
// linalg.hpp  --  Small linear-algebra helpers (dot, norm, add, sub, scale)
// Owner: Trevor (M6)
//
// WHY THIS FILE EXISTS
//   The SVM maths is full of expressions like  w.x + b,  ||w||,  w - eta*g.
//   Instead of every member writing their own loops, everyone calls these
//   functions. That means one place to fix bugs and fewer duplicated lines.
//
// ERROR RULE
//   Functions that combine two vectors throw DimensionMismatch if the sizes
//   differ. They never read out of bounds.
//
// NOTE FOR TEAMMATES
//   scale(v, k) multiplies a Vector by a number. It has NOTHING to do with
//   StandardScaler / MinMaxScaler in scaler.hpp (feature scaling, M3).
// =============================================================================
#pragma once

#include <string>
#include <vector>

#include <ML8_svm/errors.hpp>
#include <ML8_svm/types.hpp>

namespace ml8_svm {

// ---- Vector maths ----------------------------------------------------------

// Dot product:  a.b = a[0]*b[0] + a[1]*b[1] + ...
// Throws DimensionMismatch if a.size() != b.size().
// Example: dot({1,2,3}, {4,5,6}) = 4 + 10 + 18 = 32
double dot(const Vector& a, const Vector& b);

// Euclidean length:  ||v|| = sqrt(v.v)
// Example: norm({3,4}) = 5.   The norm of an empty vector is 0.
double norm(const Vector& v);

// Element-wise sum:  result[i] = a[i] + b[i]
// Throws DimensionMismatch if sizes differ.
Vector add(const Vector& a, const Vector& b);

// Element-wise difference:  result[i] = a[i] - b[i]
// Throws DimensionMismatch if sizes differ.
Vector sub(const Vector& a, const Vector& b);

// Multiply every element by the number k:  result[i] = k * v[i]
Vector scale(const Vector& v, double k);

// ---- Shape checks (used by many modules to validate their input) -----------

// Throws DimensionMismatch if a.size() != b.size().
// 'where' is the name of the calling function; it goes in the error message,
// e.g. check_same_size(w, x, "Hyperplane::score").
void check_same_size(const Vector& a, const Vector& b, const std::string& where);

// Checks that a feature matrix X and its label vector y are consistent:
//   - X is not empty                     (else InvalidArgument)
//   - X.size() == y.size()               (else DimensionMismatch)
//   - every row of X has the same length (else DimensionMismatch)
// Call this at the top of any function that takes (X, y).
void check_X_y(const Matrix& X, const std::vector<int>& y, const std::string& where);

}  // namespace ml8_svm
