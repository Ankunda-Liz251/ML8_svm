#include <ML8_svm/support_vectors.hpp>

namespace ml8_svm {

// STUB (Week 1): compiles and links so other modules can build against it.
// Week 2: implement the check y_i * h.score(x_i) <= 1 + tol, with input
// validation (see the contract in support_vectors.hpp).
std::vector<std::size_t> find_support_vectors(const Hyperplane& /*h*/,
                                              const Matrix& /*X*/,
                                              const std::vector<int>& /*y*/,
                                              double /*tol*/) {
    return {};
}

}  // namespace ml8_svm