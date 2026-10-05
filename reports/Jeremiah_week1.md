1. Completed
This week I designed the interface of the preprocessing module and created the skeleton so the rest of the team can build against it.
•Declared encode_binary_labels, impute_missing, shuffle and validate in preprocessing.hpp.
•Declared the Split struct and train_test_split (seeded, stratified).
•Created stub implementations in preprocessing.cpp that compile but do nothing yet.
•Created the skeleton test file tests/test_preprocessing.cpp.

2. In progress
•Agreeing the label rule: the smaller class value maps to -1 and the larger to +1.
•Agreeing with M1 that empty CSV fields load as NaN, so impute_missing has missing values to handle.
•Planning test cases for Week 2, including invalid inputs and seeded reproducibility.

3. Challenges and blockers
•The module depends on Dataset (M1) and errors.hpp (M6, due Tuesday 6 October). Until they are merged, I code against the stubs.

4. Next week (Week 2: 12 - 18 October)
•Implement all five functions in preprocessing.cpp.
•Write unit tests, including invalid inputs and the stratified split.
•Make at least 3 commits on the feature branch and review M3's pull requests.

5. Code summary

Header (include/ML8_svm/preprocessing.hpp):
#pragma once
#include <ML8_svm/dataset.hpp>
#include <ML8_svm/errors.hpp>
namespace ml8_svm {
void encode_binary_labels(Dataset&);   // -> {-1,+1}
void impute_missing(Dataset&);         // column mean / drop row
void shuffle(Dataset&, unsigned seed);
void validate(const Dataset&);         // throws on bad data
struct Split { Dataset train, test; };
Split train_test_split(const Dataset&, double test_ratio, unsigned seed);
}
Stubs (src/preprocessing.cpp):
#include <ML8_svm/preprocessing.hpp>
namespace ml8_svm 
void encode_binary_labels(Dataset&) {}
void impute_missing(Dataset&) {}
void shuffle(Dataset&, unsigned) {}
void validate(const Dataset&) {}
Split train_test_split(const Dataset& d, double, unsigned) { return {d, d}; }
}
