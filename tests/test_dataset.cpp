// Tests for load_csv (ML8_svm/dataset.hpp).
//
// Run the program: it prints "FAILED: ..." for every check that fails and a summary at the end.
// The exit code is 0 when everything passed, so CTest can use it.

#include <cmath>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>

#include <ML8_svm/dataset.hpp>
#include <ML8_svm/errors.hpp>

using namespace ml8_svm;

// Folder that contains input/. The build system can set it; this default works
// when the program is run from the project's main folder.
#ifndef ML8_DATA_DIR
#define ML8_DATA_DIR "data"
#endif

int total_checks = 0;
int failed_checks = 0;

// Record one check. If it is false, print what went wrong.
void check(bool condition, const std::string& description) {
    total_checks++;
    if (!condition) {
        failed_checks++;
        std::cout << "FAILED: " << description << "\n";
    }
}

// True if two decimal numbers are equal (apart from tiny rounding errors).
bool nearly_equal(double a, double b) {
    return std::fabs(a - b) < 0.000001;
}

// Small CSV files for the tests are written to this file and removed afterwards.
const std::string TEMP_FILE = "temp_test_dataset.csv";

void write_file(const std::string& name, const std::string& text) {
    std::ofstream out(name, std::ios::binary);
    out << text;
}

// True if load_csv throws InvalidArgument for this file.
bool load_fails_with_invalid(const std::string& path, bool has_header = true, int label_col = -1) {
    try {
        load_csv(path, has_header, label_col);
    } catch (const InvalidArgument&) {
        return true;
    } catch (...) {
        return false;  // some other kind of error
    }
    return false;  // no error at all
}

//  the two bundled datasets

void test_linear_separable() {
    Dataset d = load_csv(std::string(ML8_DATA_DIR) + "/input/linear_separable.csv");
    check(d.n_samples() == 200, "linear_separable has 200 samples");
    check(d.n_features() == 2, "linear_separable has 2 features");
    check(d.y.size() == 200, "linear_separable has 200 labels");
    check(d.feature_names.size() == 2 && d.feature_names[0] == "x1" && d.feature_names[1] == "x2",
          "linear_separable feature names are x1, x2");
    check(nearly_equal(d.X[0][0], 2.1935) && nearly_equal(d.X[0][1], 3.6692), "first row of linear_separable");
    check(d.y[0] == 1, "first label of linear_separable is 1");

    int positive = 0;
    int negative = 0;
    for (size_t i = 0; i < d.y.size(); i++) {
        if (d.y[i] == 1) positive++;
        if (d.y[i] == -1) negative++;
    }
    check(positive == 100, "100 samples with label +1");
    check(negative == 100, "100 samples with label -1");
}

void test_iris() {
    Dataset d = load_csv(std::string(ML8_DATA_DIR) + "/input/iris_versicolor_virginica.csv");
    check(d.n_samples() == 100, "iris has 100 samples");
    check(d.n_features() == 4, "iris has 4 features");
    check(d.feature_names.size() == 4 && d.feature_names[0] == "sepal_length" && d.feature_names[3] == "petal_width",
          "iris feature names");
    check(nearly_equal(d.X[0][0], 7.0) && nearly_equal(d.X[0][1], 3.2) && nearly_equal(d.X[0][2], 4.7)
              && nearly_equal(d.X[0][3], 1.4),
          "first row of iris");
    check(d.y[0] == 0, "first iris label is 0");
    check(d.y[99] == 1, "last iris label is 1");

    int zeros = 0;
    int ones = 0;
    for (size_t i = 0; i < d.y.size(); i++) {
        if (d.y[i] == 0) zeros++;
        if (d.y[i] == 1) ones++;
    }
    check(zeros == 50 && ones == 50, "iris has 50 samples of each class");
}

int main() {
    try {
        test_linear_separable();
        test_iris();
    } catch (const std::exception& e) {
        std::cout << "Unexpected error: " << e.what() << "\n";
        return 1;
    }

    std::cout << "test_dataset: " << (total_checks - failed_checks) << " of " << total_checks << " checks passed\n";
    if (failed_checks == 0) {
        return 0;
    }
    return 1;
}
