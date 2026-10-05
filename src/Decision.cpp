//decision.cpp
//  The functions exist so the library compiles and others can link against
//  them, but the real work is written in Week 2.#include <ML8_svm/decision.hpp>
#include <ML8_svm/errors.hpp>

namespace ml8_svm {

double decision_function(const Hyperplane&, const Vector&) {
    throw SvmError("decision_function: not implemented yet (Week 2)");
}

Vector decision_function(const Hyperplane&, const Matrix&) {
    throw SvmError("decision_function: not implemented yet (Week 2)");
}

int to_label(double) {
    throw SvmError("to_label: not implemented yet (Week 2)");
}

} // namespace ml8_svm
