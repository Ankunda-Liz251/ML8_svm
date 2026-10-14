#pragma once
#include <ML8_svm/dataset.hpp>

namespace ml8_svm {

void encode_binary_labels(Dataset&);   // any two classes -> {-1,+1}
void impute_missing(Dataset&);         // NaN -> column mean
void shuffle(Dataset&, unsigned seed); // seeded, X and y stay paired
void validate(const Dataset&);         // throws if dataset is unusable

struct Split { Dataset train, test; };
Split train_test_split(const Dataset&, double test_ratio, unsigned seed);

} // namespace ml8_svm