#include <ML8_svm/decision.hpp>
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

} 
