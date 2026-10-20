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

// ------------------------------------------------------------ the two bundled datasets

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

// ------------------------------------------------------------ loader options

void test_no_header() {
    write_file(TEMP_FILE, "1,2,0\n3,4,1\n");
    Dataset d = load_csv(TEMP_FILE, false);  // false = the file has no header line
    check(d.n_samples() == 2, "no header: 2 samples");
    check(d.feature_names.size() == 2 && d.feature_names[0] == "x0" && d.feature_names[1] == "x1",
          "no header: names are made up (x0, x1)");
    check(nearly_equal(d.X[1][0], 3.0) && nearly_equal(d.X[1][1], 4.0), "no header: second row");
    check(d.y[0] == 0 && d.y[1] == 1, "no header: labels");
    std::remove(TEMP_FILE.c_str());
}

void test_semicolon_and_label_in_first_column() {
    write_file(TEMP_FILE, "label;a;b\n0;1.5;2.5\n1;3.5;4.5\n");
    Dataset d = load_csv(TEMP_FILE, true, 0, ';');  // label is column 0, fields are separated by ';'
    check(d.feature_names.size() == 2 && d.feature_names[0] == "a" && d.feature_names[1] == "b",
          "label first: feature names are a, b");
    check(nearly_equal(d.X[0][0], 1.5) && nearly_equal(d.X[0][1], 2.5), "label first: first row");
    check(d.y[0] == 0 && d.y[1] == 1, "label first: labels");
    std::remove(TEMP_FILE.c_str());
}

void test_negative_label_index() {
    write_file(TEMP_FILE, "a,l,b\n1,0,5\n2,1,6\n");
    Dataset d = load_csv(TEMP_FILE, true, -2);  // -2 = the second column from the end
    check(d.feature_names.size() == 2 && d.feature_names[0] == "a" && d.feature_names[1] == "b",
          "label_col -2: feature names are a, b");
    check(nearly_equal(d.X[0][0], 1.0) && nearly_equal(d.X[0][1], 5.0), "label_col -2: first row");
    check(d.y[0] == 0 && d.y[1] == 1, "label_col -2: labels");
    std::remove(TEMP_FILE.c_str());
}

void test_signed_labels() {
    write_file(TEMP_FILE, "a,label\n1.5,+1\n2.5,-1\n");
    Dataset d = load_csv(TEMP_FILE);
    check(d.y.size() == 2 && d.y[0] == 1 && d.y[1] == -1, "labels +1 and -1 are accepted");
    std::remove(TEMP_FILE.c_str());
}

// ------------------------------------------------------------ error cases

void test_missing_file() {
    bool got_file_error = false;
    try {
        load_csv("/no/such/folder/nothing.csv");
    } catch (const FileError&) {
        got_file_error = true;
    } catch (...) {
    }
    check(got_file_error, "a file that does not exist gives FileError");
}

void test_empty_file() {
    write_file(TEMP_FILE, "");
    check(load_fails_with_invalid(TEMP_FILE), "empty file gives InvalidArgument");
    write_file(TEMP_FILE, "\n  \n\n");
    check(load_fails_with_invalid(TEMP_FILE), "file with only blank lines gives InvalidArgument");
    write_file(TEMP_FILE, "a,b,label\n");
    check(load_fails_with_invalid(TEMP_FILE), "header without data rows gives InvalidArgument");
    std::remove(TEMP_FILE.c_str());
}

void test_wrong_number_of_fields() {
    write_file(TEMP_FILE, "a,b,label\n1,2,0\n3,4\n");
    check(load_fails_with_invalid(TEMP_FILE), "row with too few fields gives InvalidArgument");
    write_file(TEMP_FILE, "a,b,label\n1,2,0\n3,4,1,9\n");
    check(load_fails_with_invalid(TEMP_FILE), "row with too many fields gives InvalidArgument");
    std::remove(TEMP_FILE.c_str());
}

void test_not_a_number() {
    write_file(TEMP_FILE, "a,b,label\n1,abc,0\n");
    check(load_fails_with_invalid(TEMP_FILE), "text in a feature column gives InvalidArgument");
    write_file(TEMP_FILE, "a,b,label\n1,2x,0\n");
    check(load_fails_with_invalid(TEMP_FILE), "'2x' is not a number");
    std::remove(TEMP_FILE.c_str());
}

void test_bad_labels() {
    write_file(TEMP_FILE, "a,b,label\n1,2,1.5\n");
    check(load_fails_with_invalid(TEMP_FILE), "label 1.5 is not an integer");
    write_file(TEMP_FILE, "a,b,label\n1,2,x\n");
    check(load_fails_with_invalid(TEMP_FILE), "label x is not an integer");
    write_file(TEMP_FILE, "a,b,label\n1,2,\n");
    check(load_fails_with_invalid(TEMP_FILE), "an empty label is not allowed");
    std::remove(TEMP_FILE.c_str());
}

void test_label_column_out_of_range() {
    write_file(TEMP_FILE, "a,b,label\n1,2,0\n");
    check(load_fails_with_invalid(TEMP_FILE, true, 3), "label_col 3 does not exist in a 3-column file");
    check(load_fails_with_invalid(TEMP_FILE, true, -4), "label_col -4 does not exist in a 3-column file");
    std::remove(TEMP_FILE.c_str());
}

void test_one_column_only() {
    write_file(TEMP_FILE, "a\n1\n");
    check(load_fails_with_invalid(TEMP_FILE), "a file with one column gives InvalidArgument");
    std::remove(TEMP_FILE.c_str());
}

void test_error_message_has_line_number() {
    write_file(TEMP_FILE, "a,b,label\n1,2,0\n3,4\n");  // the bad row is on line 3
    std::string message = "";
    try {
        load_csv(TEMP_FILE);
    } catch (const InvalidArgument& e) {
        message = e.what();
    }
    check(message.find("line 3") != std::string::npos, "the error message names line 3");
    std::remove(TEMP_FILE.c_str());
}

// ------------------------------------------------------------ valid files with odd details

void test_missing_values() {
    write_file(TEMP_FILE, "a,b,label\n1,,0\nNaN,2,1\nNA,?,0\nnan,5,1\n");
    Dataset d = load_csv(TEMP_FILE);
    check(d.n_samples() == 4, "missing values: 4 samples");
    check(nearly_equal(d.X[0][0], 1.0), "missing values: normal value is kept");
    check(std::isnan(d.X[0][1]), "an empty field becomes NaN");
    check(std::isnan(d.X[1][0]), "NaN becomes NaN");
    check(nearly_equal(d.X[1][1], 2.0), "missing values: normal value is kept (2)");
    check(std::isnan(d.X[2][0]), "NA becomes NaN");
    check(std::isnan(d.X[2][1]), "? becomes NaN");
    check(std::isnan(d.X[3][0]), "nan becomes NaN");
    check(d.y[0] == 0 && d.y[1] == 1 && d.y[2] == 0 && d.y[3] == 1, "missing values: labels are fine");
    std::remove(TEMP_FILE.c_str());
}

void test_spaces_blank_lines_and_windows_line_endings() {
    write_file(TEMP_FILE, "a, b, label\r\n\r\n 1 , 2 , 1 \r\n\r\n3,4,0\r\n");
    Dataset d = load_csv(TEMP_FILE);
    check(d.feature_names.size() == 2 && d.feature_names[0] == "a" && d.feature_names[1] == "b",
          "spaces around names are removed");
    check(d.n_samples() == 2, "blank lines are skipped");
    check(nearly_equal(d.X[0][0], 1.0) && nearly_equal(d.X[0][1], 2.0), "spaces around numbers are removed");
    check(d.y[0] == 1 && d.y[1] == 0, "labels with spaces and \\r");
    std::remove(TEMP_FILE.c_str());
}

void test_excel_byte_order_mark() {
    // "\xEF\xBB\xBF" is the BOM. The text is split in two pieces so that the 'a'
    // after it is not read as part of the \x escape.
    write_file(TEMP_FILE, "\xEF\xBB\xBF" "a,b,label\n1,2,0\n");
    Dataset d = load_csv(TEMP_FILE);
    check(d.feature_names.size() == 2 && d.feature_names[0] == "a", "BOM is not part of the first name");

    write_file(TEMP_FILE, "\xEF\xBB\xBF" "1,2,0\n");
    Dataset e = load_csv(TEMP_FILE, false);
    check(e.n_samples() == 1 && nearly_equal(e.X[0][0], 1.0), "BOM is skipped when there is no header");
    std::remove(TEMP_FILE.c_str());
}

// ------------------------------------------------------------ number formats and file layouts

void test_number_formats() {
    write_file(TEMP_FILE, "a,b,label\n-2.5,1e-3,0\n+3,.5,1\n");
    Dataset d = load_csv(TEMP_FILE);
    check(nearly_equal(d.X[0][0], -2.5), "negative number");
    check(nearly_equal(d.X[0][1], 0.001), "scientific notation 1e-3");
    check(nearly_equal(d.X[1][0], 3.0), "number with a plus sign");
    check(nearly_equal(d.X[1][1], 0.5), "number without a leading zero (.5)");
    std::remove(TEMP_FILE.c_str());
}

void test_tab_delimiter() {
    write_file(TEMP_FILE, "a\tb\tlabel\n1\t2\t0\n3\t4\t1\n");
    Dataset d = load_csv(TEMP_FILE, true, -1, '\t');
    check(d.n_samples() == 2, "tab-separated file: 2 samples");
    check(d.feature_names.size() == 2 && d.feature_names[1] == "b", "tab-separated file: feature names");
    check(nearly_equal(d.X[1][0], 3.0) && d.y[1] == 1, "tab-separated file: second row");
    std::remove(TEMP_FILE.c_str());
}

void test_no_final_newline() {
    write_file(TEMP_FILE, "a,b,label\n1,2,0\n3,4,1");  // the last line has no \n
    Dataset d = load_csv(TEMP_FILE);
    check(d.n_samples() == 2, "a file without a final newline still gives 2 samples");
    check(d.y[1] == 1, "the last line is read correctly");
    std::remove(TEMP_FILE.c_str());
}

void test_names_with_spaces() {
    write_file(TEMP_FILE, "sepal length, petal width ,label\n1,2,0\n");
    Dataset d = load_csv(TEMP_FILE);
    check(d.feature_names.size() == 2, "two feature names");
    check(d.feature_names[0] == "sepal length", "a space inside a name is kept");
    check(d.feature_names[1] == "petal width", "spaces around a name are removed");
    std::remove(TEMP_FILE.c_str());
}

// ------------------------------------------------------------ quoted fields

void test_quoted_fields() {
    write_file(TEMP_FILE, "\"x1\",\"x2\",\"label\"\n\"1.5\",\"2.5\",\"0\"\n\"3.5\",4.5,1\n");
    Dataset d = load_csv(TEMP_FILE);
    check(d.feature_names.size() == 2 && d.feature_names[0] == "x1" && d.feature_names[1] == "x2",
          "quotes around names are removed");
    check(d.n_samples() == 2, "quoted file: 2 samples");
    check(nearly_equal(d.X[0][0], 1.5) && nearly_equal(d.X[0][1], 2.5), "quotes around numbers are removed");
    check(d.y[0] == 0 && d.y[1] == 1, "quotes around labels are removed");
    check(nearly_equal(d.X[1][0], 3.5) && nearly_equal(d.X[1][1], 4.5), "quoted and unquoted fields can be mixed");
    std::remove(TEMP_FILE.c_str());
}

void test_spaces_inside_quotes() {
    write_file(TEMP_FILE, "\" a \", b ,label\n\" 1 \",2,0\n");
    Dataset d = load_csv(TEMP_FILE);
    check(d.feature_names.size() == 2 && d.feature_names[0] == "a" && d.feature_names[1] == "b",
          "spaces inside and outside quotes are removed");
    check(nearly_equal(d.X[0][0], 1.0), "a quoted number with spaces is read");
    std::remove(TEMP_FILE.c_str());
}

void test_quote_that_is_not_closed() {
    write_file(TEMP_FILE, "a,b,label\n\"1,2,0\n");
    check(load_fails_with_invalid(TEMP_FILE), "a field with only one quote is not a number");
    std::remove(TEMP_FILE.c_str());
}

// ------------------------------------------------------------ labels written like 1.0

void test_labels_written_with_decimals() {
    write_file(TEMP_FILE, "a,label\n1.5,1.0\n2.5,-1.0\n3.5,0\n");
    Dataset d = load_csv(TEMP_FILE);
    check(d.y.size() == 3, "3 labels");
    check(d.y[0] == 1 && d.y[1] == -1 && d.y[2] == 0, "labels 1.0, -1.0 and 0 become 1, -1 and 0");
    std::remove(TEMP_FILE.c_str());
}

void test_labels_that_are_still_rejected() {
    write_file(TEMP_FILE, "a,label\n1,1.5\n");
    check(load_fails_with_invalid(TEMP_FILE), "label 1.5 is still rejected");
    write_file(TEMP_FILE, "a,label\n1,nan\n");
    check(load_fails_with_invalid(TEMP_FILE), "label nan is rejected");
    write_file(TEMP_FILE, "a,label\n1,inf\n");
    check(load_fails_with_invalid(TEMP_FILE), "label inf is rejected");
    std::remove(TEMP_FILE.c_str());
}

int main() {
    try {
        test_linear_separable();
        test_iris();
        test_no_header();
        test_semicolon_and_label_in_first_column();
        test_negative_label_index();
        test_signed_labels();
        test_missing_file();
        test_empty_file();
        test_wrong_number_of_fields();
        test_not_a_number();
        test_bad_labels();
        test_label_column_out_of_range();
        test_one_column_only();
        test_error_message_has_line_number();
        test_missing_values();
        test_spaces_blank_lines_and_windows_line_endings();
        test_excel_byte_order_mark();
        test_number_formats();
        test_tab_delimiter();
        test_no_final_newline();
        test_names_with_spaces();
        test_quoted_fields();
        test_spaces_inside_quotes();
        test_quote_that_is_not_closed();
        test_labels_written_with_decimals();
        test_labels_that_are_still_rejected();
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
