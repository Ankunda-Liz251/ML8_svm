// =============================================================================
// linalg.cpp  --  Implementation of the helpers declared in linalg.hpp
// Owner: Trevor (M6)
// =============================================================================
#include <ML8_svm/linalg.hpp>

#include <cmath>  // std::sqrt

namespace ml8_svm {

void check_same_size(const Vector& a, const Vector& b, const std::string& where) {
    if (a.size() != b.size()) {
        // Put both sizes in the message so the bug is easy to find.
        throw DimensionMismatch(where + ": size mismatch (" +
                                std::to_string(a.size()) + " vs " +
                                std::to_string(b.size()) + ")");
    }
}

void check_X_y(const Matrix& X, const std::vector<int>& y, const std::string& where) {
    if (X.empty()) {
        throw InvalidArgument(where + ": X is empty");
    }
    if (X.size() != y.size()) {
        throw DimensionMismatch(where + ": X has " + std::to_string(X.size()) +
                                " rows but y has " + std::to_string(y.size()) +
                                " labels");
    }
    // Every sample must have the same number of features as the first one.
    const std::size_t d = X[0].size();
    for (std::size_t i = 1; i < X.size(); ++i) {
        if (X[i].size() != d) {
            throw DimensionMismatch(where + ": row " + std::to_string(i) +
                                    " has " + std::to_string(X[i].size()) +
                                    " features, expected " + std::to_string(d));
        }
    }
}

double dot(const Vector& a, const Vector& b) {
    check_same_size(a, b, "dot");
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];  // multiply matching entries and accumulate
    }
    return sum;
}

double norm(const Vector& v) {
    // ||v|| = sqrt(v.v). dot(v, v) cannot throw because both sizes are equal.
    return std::sqrt(dot(v, v));
}

Vector add(const Vector& a, const Vector& b) {
    check_same_size(a, b, "add");
    Vector result(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) result[i] = a[i] + b[i];
    return result;
}

Vector sub(const Vector& a, const Vector& b) {
    check_same_size(a, b, "sub");
    Vector result(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) result[i] = a[i] - b[i];
    return result;
}

Vector scale(const Vector& v, double k) {
    Vector result(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) result[i] = k * v[i];
    return result;
}

}  // namespace ml8_svm
