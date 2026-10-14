#include <ML8_svm/preprocessing.hpp>
#include <stdexcept>

namespace ml8_svm {

namespace {
[[noreturn]] void todo() { throw std::logic_error("preprocessing: not implemented"); }
}

void encode_binary_labels(Dataset&) { todo(); }
void impute_missing(Dataset&) { todo(); }
void shuffle(Dataset&, unsigned) { todo(); }
void validate(const Dataset&) { todo(); }
Split train_test_split(const Dataset&, double, unsigned) { todo(); }

} // namespace ml8_svm