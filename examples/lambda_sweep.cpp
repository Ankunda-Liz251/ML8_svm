// Lambda sweep: trains one linear SVM for each regularization strength (lambda)
// and records how the model changes. Results are printed and saved as a CSV file.
//
// Usage (run from the repository root):
//   lambda_sweep [input_csv] [output_csv]
// Defaults: data/input/linear_separable.csv -> data/output/lambda_sweep.csv
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <ML8_svm/dataset.hpp>
#include <ML8_svm/errors.hpp>
#include <ML8_svm/linalg.hpp>
#include <ML8_svm/linear_svm.hpp>
#include <ML8_svm/optimizer.hpp>
#include <ML8_svm/preprocessing.hpp>
#include <ML8_svm/regularization.hpp>
#include <ML8_svm/scaler.hpp>

using namespace std;
using namespace ml8_svm;

// Fraction of predictions that equal the true labels.
static double accuracy_of(const vector<int>& truth, const vector<int>& predicted) {
    size_t correct = 0;
    for (size_t i = 0; i < truth.size(); ++i) {
        if (truth[i] == predicted[i]) ++correct;
    }
    return static_cast<double>(correct) / truth.size();
}

int main(int argc, char* argv[]) {
    const string input_path = argc > 1 ? argv[1] : "data/input/linear_separable.csv";
    const string output_path = argc > 2 ? argv[2] : "data/output/lambda_sweep.csv";

    try {
        // Load the data, encode labels as -1/+1 and hold out 20% for testing.
        Dataset data = load_csv(input_path);
        encode_binary_labels(data);
        validate(data);
        Split split = train_test_split(data, 0.2, 42);

        // Scale features. The scaler learns its statistics from the training set only.
        StandardScaler scaler;
        Matrix X_train = scaler.fit_transform(split.train.X);
        Matrix X_test = scaler.transform(split.test.X);

        // Same optimizer settings for every lambda, so only lambda changes.
        OptimizerConfig config;
        config.learning_rate = 0.1;
        config.epochs = 500;
        config.batch_size = X_train.size();  // full batch
        config.tol = 1e-6;
        config.decay = true;
        config.seed = 42;

        // 11 values spaced evenly on a log scale: 1e-4, 10^-3.5, ..., 10.
        vector<double> lambdas;
        for (int k = 0; k <= 10; ++k) lambdas.push_back(pow(10.0, -4.0 + 0.5 * k));

        ofstream out(output_path);
        if (!out) throw FileError("cannot open output file: " + output_path);
        out << "lambda,weight_norm,margin_width,objective,"
               "train_accuracy,test_accuracy,support_vectors\n";
        out << setprecision(8);

        cout << fixed << setprecision(4);
        cout << setw(10) << "lambda" << setw(12) << "||w||" << setw(12) << "margin"
             << setw(12) << "objective" << setw(10) << "train" << setw(10) << "test"
             << setw(8) << "SVs" << '\n';

        for (double lambda : lambdas) {
            LinearSVM svm(lambda, config);
            svm.fit(X_train, split.train.y);

            const Hyperplane& h = svm.hyperplane();
            // ml8_svm:: is written out because std::norm also exists.
            const double weight_norm = ml8_svm::norm(h.w);
            const double margin = h.margin_width();
            const double obj = objective(h, X_train, split.train.y, lambda);
            const double train_acc = accuracy_of(split.train.y, svm.predict(X_train));
            const double test_acc = accuracy_of(split.test.y, svm.predict(X_test));
            const size_t n_sv = svm.support_vector_indices().size();

            out << lambda << ',' << weight_norm << ',' << margin << ',' << obj << ','
                << train_acc << ',' << test_acc << ',' << n_sv << '\n';

            cout << setw(10) << lambda << setw(12) << weight_norm << setw(12) << margin
                 << setw(12) << obj << setw(10) << train_acc << setw(10) << test_acc
                 << setw(8) << n_sv << '\n';
        }

        cout << "\nSaved results to " << output_path << '\n';
    } catch (const SvmError& e) {
        cerr << "Error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
