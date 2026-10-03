#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <ML8_svm/types.hpp>

namespace ml8_svm {

// A labelled dataset.
// Row i of X is one sample and y[i] is its label. After loading, labels are
// whatever integers the file contained (for example 0/1 or -1/+1); use
// encode_binary_labels() from preprocessing.hpp to turn them into -1/+1.
struct Dataset {
    std::vector<std::vector<double>> X;      //< n_samples x n_features
    std::vector<int> y;                      //< n_samples labels
    std::vector<std::string> feature_names;  //< n_features names ("x0", "x1", ... if the file has no header)

    // Number of samples (rows of X).
    std::size_t n_samples() const { return X.size(); }

    // Number of features (columns of X); 0 for an empty dataset.
    std::size_t n_features() const { return X.empty() ? 0 : X.front().size(); }
};

// Load a numeric CSV file into a Dataset.
//
// @param path       Path of the CSV file.
// @param has_header true if the first line holds column names.
// @param label_col  Index of the label column. A negative value counts from
//                   the end, so -1 (default) means the last column.
// @param delim      Field delimiter (default ',').
//
// File format:
//  - Every feature field is a number. An empty field or one of the tokens
//    NaN, nan, NA or ? is read as a missing value and stored as
//    std::numeric_limits<double>::quiet_NaN() (see impute_missing()).
//  - The label field must be an integer (for example 0, 1, -1 or +1).
//  - Blank lines are ignored; leading and trailing spaces in a field are trimmed.
//
// @throws FileError        if the file cannot be opened.
// @throws InvalidArgument  if the file is empty, a row has a different number
//                          of fields from the first row, a field that is not a
//                          missing-value token is not a number, a label is not
//                          an integer, or label_col is out of range.
Dataset load_csv(const std::string& path,
                 bool has_header = true,
                 int label_col = -1,
                 char delim = ',');

}  // namespace ml8_svm
