// Tests for ML8_svm/dataset.hpp.
//
// Week 1: skeleton only. It checks that the public interface matches the
// agreed contract. The real checks are added in Week 2 once load_csv exists.
//
// Planned test cases (Week 2):
//   1. linear_separable.csv loads: 200 samples, 2 features, labels in {-1, +1},
//      feature_names == {"x1", "x2"}.
//   2. iris_versicolor_virginica.csv loads: 100 samples, 4 features, labels {0, 1}.
//   3. has_header = false: feature names are generated ("x0", "x1", ...).
//   4. A different delimiter (for example ';') and a label column that is not
//      the last one (label_col = 0, and a negative index).
//   5. Missing values (empty field, NaN, NA, ?) are read as NaN.
//   6. A file that does not exist throws FileError.
//   7. A ragged row, a non-numeric field, a non-integer label, an empty file
//      and an out-of-range label_col each throw InvalidArgument.
//
// Data files are found through the ML8_DATA_DIR compile definition, which is
// set in tests/CMakeLists.txt (ask M3), for example:
//   const std::string path = std::string(ML8_DATA_DIR) + "/input/linear_separable.csv";

#include <iostream>
#include <string>

#include <ML8_svm/dataset.hpp>

int main() {
    // Compile-time check of the public API signature.
    ml8_svm::Dataset (*fn)(const std::string&, bool, int, char) = &ml8_svm::load_csv;
    (void)fn;

    ml8_svm::Dataset empty;
    if (empty.n_samples() != 0 || empty.n_features() != 0) {
        std::cerr << "test_dataset: an empty Dataset must report 0 samples and 0 features\n";
        return 1;
    }

    std::cout << "test_dataset: skeleton OK (real checks come in Week 2)\n";
    return 0;
}
