#include <ML8_svm/hyperplane.hpp>

namespace ml8_svm {

namespace {

// Below this, ||w|| is treated as zero (degenerate hyperplane).
constexpr double kZeroNormEps = 1e-12;

// Returns ||w||, throwing if the hyperplane is degenerate (w == 0).
double nonzero_norm(const Vector& w, const char* caller) {
    const double n = norm(w);
    if (n < kZeroNormEps) {
        throw InvalidArgument(std::string("Hyperplane::") + caller +
                              ": weight vector is zero");
    }
    return n;
}

void check_label(int y, const char* caller) {
    if (y != -1 && y != +1) {
        throw InvalidArgument(std::string("Hyperplane::") + caller +
                              ": label must be -1 or +1");
    }
}

} // namespace

double Hyperplane::score(const Vector& x) const {
    if (x.size() != w.size()) {
        throw DimensionMismatch("Hyperplane::score: x has " +
                                std::to_string(x.size()) + " features, w has " +
                                std::to_string(w.size()));
    }
    return dot(w, x) + b;
}

double Hyperplane::distance(const Vector& x) const {
    const double n = nonzero_norm(w, "distance");
    return score(x) / n;
}

double Hyperplane::functional_margin(const Vector& x, int y) const {
    check_label(y, "functional_margin");
    return static_cast<double>(y) * score(x);
}

double Hyperplane::geometric_margin(const Vector& x, int y) const {
    check_label(y, "geometric_margin");
    return static_cast<double>(y) * distance(x);  // reuses distance(): no duplicated code
}

double Hyperplane::margin_width() const {
    return 2.0 / nonzero_norm(w, "margin_width");
}

} // namespace ml8_svm