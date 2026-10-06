#include <ML8_svm/hyperplane.hpp>

namespace ml8_svm {

// Week 1: stub implementations so the library links.
// Week 2: replace each TODO with the real logic (see reports/week-01.md notes).

double Hyperplane::score(const Vector& x) const {
    (void)x;
    // TODO (week 2): check x.size() == w.size(), return dot(w, x) + b.
    return 0.0;
}

double Hyperplane::distance(const Vector& x) const {
    (void)x;
    // TODO (week 2): throw InvalidArgument if ||w|| == 0, return score(x) / norm(w).
    return 0.0;
}

double Hyperplane::functional_margin(const Vector& x, int y) const {
    (void)x;
    (void)y;
    // TODO (week 2): validate y in {-1,+1}, return y * score(x).
    return 0.0;
}

double Hyperplane::geometric_margin(const Vector& x, int y) const {
    (void)x;
    (void)y;
    // TODO (week 2): validate y and w != 0, return y * score(x) / norm(w).
    return 0.0;
}

double Hyperplane::margin_width() const {
    // TODO (week 2): throw InvalidArgument if ||w|| == 0, return 2.0 / norm(w).
    return 0.0;
}

} // namespace ml8_svm