// Example 1: load a CSV file and print a summary of the data.
//
// Usage:    example1_load_csv [path-to-csv-file]
// Example:  example1_load_csv data/input/linear_separable.csv
//
// If you give no file name, data/input/linear_separable.csv is used.

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <ML8_svm/dataset.hpp>
#include <ML8_svm/errors.hpp>

using namespace ml8_svm;

int main(int argc, char* argv[]) {
    // Which file do we load? The first argument, or a default file.
    std::string path = "data/input/linear_separable.csv";
    if (argc >= 2) {
        path = argv[1];
    }

    // Step 1: load the file. load_csv throws an error when something is wrong, so we catch it.
    Dataset data;
    try {
        data = load_csv(path);
    } catch (const FileError& e) {
        std::cout << "Could not open the file: " << e.what() << "\n";
        return 1;
    } catch (const InvalidArgument& e) {
        std::cout << "The file is not a valid dataset: " << e.what() << "\n";
        return 1;
    }

    // Step 2: the size of the dataset.
    std::cout << "File:     " << path << "\n";
    std::cout << "Samples:  " << data.n_samples() << "\n";
    std::cout << "Features: " << data.n_features() << "\n";

    // Step 3: how many samples have each label.
    std::vector<int> labels;  // the different labels we have seen
    std::vector<int> counts;  // counts[k] = number of samples with label labels[k]
    for (size_t i = 0; i < data.y.size(); i++) {
        bool found = false;
        for (size_t k = 0; k < labels.size(); k++) {
            if (labels[k] == data.y[i]) {
                counts[k]++;
                found = true;
            }
        }
        if (!found) {
            labels.push_back(data.y[i]);
            counts.push_back(1);
        }
    }
    std::cout << "\nLabels:\n";
    for (size_t k = 0; k < labels.size(); k++) {
        std::cout << "  label " << labels[k] << ": " << counts[k] << " samples\n";
    }

    // Step 4: for each feature, the smallest, largest and average value.
    // Missing values are loaded as NaN. We count them and leave them out of the numbers.
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\nFeatures:\n";
    int total_missing = 0;
    for (size_t j = 0; j < data.n_features(); j++) {
        int missing = 0;
        int present = 0;
        double sum = 0.0;
        double smallest = 0.0;
        double largest = 0.0;

        for (size_t i = 0; i < data.n_samples(); i++) {
            double value = data.X[i][j];
            if (std::isnan(value)) {
                missing++;
            } else {
                if (present == 0 || value < smallest) {
                    smallest = value;
                }
                if (present == 0 || value > largest) {
                    largest = value;
                }
                sum = sum + value;
                present++;
            }
        }
        total_missing = total_missing + missing;

        std::cout << "  " << data.feature_names[j] << ": ";
        if (present == 0) {
            std::cout << "no values at all\n";
        } else {
            std::cout << "min " << smallest << ", max " << largest << ", mean " << sum / present
                      << ", missing " << missing << "\n";
        }
    }

    // Step 5: the first rows of the data.
    std::cout << "\nFirst rows:\n";
    for (size_t j = 0; j < data.n_features(); j++) {
        std::cout << std::setw(14) << data.feature_names[j];
    }
    std::cout << std::setw(8) << "label" << "\n";

    size_t rows_to_show = 5;
    if (data.n_samples() < rows_to_show) {
        rows_to_show = data.n_samples();
    }
    for (size_t i = 0; i < rows_to_show; i++) {
        for (size_t j = 0; j < data.n_features(); j++) {
            if (std::isnan(data.X[i][j])) {
                std::cout << std::setw(14) << "missing";
            } else {
                std::cout << std::setw(14) << data.X[i][j];
            }
        }
        std::cout << std::setw(8) << data.y[i] << "\n";
    }

    if (total_missing > 0) {
        std::cout << "\nThere are " << total_missing
                  << " missing values. Use impute_missing() from preprocessing.hpp to fill them in.\n";
    }
    return 0;
}
