// example5_full_pipeline.cpp - Full pipeline with evaluation
//
// Usage: example5_full_pipeline [input.csv] [output_predictions.csv]
// Defaults: data/input/linear_separable.csv  data/output/predictions.csv
#include <ML8_svm/ML8_svm.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace ml8_svm;
namespace fs = std::filesystem;

static OptimizerConfig make_config(unsigned seed) {
    OptimizerConfig cfg{};
    cfg.learning_rate = 0.1;
    cfg.epochs = 500;
    cfg.batch_size = 32;
    cfg.tol = 1e-6;
    cfg.decay = true;
    cfg.seed = seed;
    return cfg;
}

static void save_predictions(const std::string& path,
                             const std::vector<int>& y_true,
                             const std::vector<int>& y_pred,
                             const Vector& scores) {
    const fs::path p(path);
    if (p.has_parent_path()) fs::create_directories(p.parent_path());
    std::ofstream out(path);
    if (!out) throw FileError("cannot write predictions to " + path);
    out << "index,y_true,y_pred,score\n" << std::setprecision(8);
    for (std::size_t i = 0; i < y_true.size(); ++i)
        out << i << ',' << y_true[i] << ',' << y_pred[i] << ',' << scores[i]
            << '\n';
}

// k-fold cross-validated accuracy. The scaler is fitted on each fold's
// training part only, so nothing leaks from the held-out fold.
static double cross_val_accuracy(const Dataset& data, int k, double lambda,
                                 const OptimizerConfig& cfg, unsigned seed) {
    const std::size_t n = data.X.size();
    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(idx.begin(), idx.end(), rng);

    double sum = 0.0;
    for (int f = 0; f < k; ++f) {
        Matrix Xtr, Xte;
        std::vector<int> ytr, yte;
        for (std::size_t pos = 0; pos < n; ++pos) {
            const std::size_t i = idx[pos];
            if (static_cast<int>(pos % k) == f) {
                Xte.push_back(data.X[i]);
                yte.push_back(data.y[i]);
            } else {
                Xtr.push_back(data.X[i]);
                ytr.push_back(data.y[i]);
            }
        }
        StandardScaler sc;
        const Matrix Xtr_s = sc.fit_transform(Xtr);
        const Matrix Xte_s = sc.transform(Xte);
        LinearSVM m(lambda, cfg);
        m.fit(Xtr_s, ytr);
        sum += accuracy(yte, m.predict(Xte_s));
    }
    return sum / k;
}

int main(int argc, char** argv) {
    const std::string in_path =
        argc > 1 ? argv[1] : "data/input/linear_separable.csv";
    const std::string out_path =
        argc > 2 ? argv[2] : "data/output/predictions.csv";
    const unsigned seed = 42;
    const double lambda = 0.01;

    try {
        // 1. Load
        Dataset data = load_csv(in_path);
        std::cout << "Loaded " << data.X.size() << " samples, "
                  << (data.X.empty() ? 0 : data.X[0].size())
                  << " features from " << in_path << "\n";

        // 2. Preprocess
        encode_binary_labels(data);
        impute_missing(data);
        validate(data);
        Split s = train_test_split(data, 0.25, seed);
        std::cout << "Train: " << s.train.X.size()
                  << " samples, Test: " << s.test.X.size() << " samples\n";

        // 3. Scale (fit on training data only)
        StandardScaler scaler;
        const Matrix Xtr = scaler.fit_transform(s.train.X);
        const Matrix Xte = scaler.transform(s.test.X);

        // 4. Fit
        const OptimizerConfig cfg = make_config(seed);
        LinearSVM model(lambda, cfg);
        model.fit(Xtr, s.train.y);
        std::cout << "Training: " << model.history().epochs_run
                  << " epochs, converged = " << std::boolalpha
                  << model.history().converged << "\n";

        // 5. Decision function + predict
        const Vector scores = model.decision_function(Xte);
        const std::vector<int> y_pred = model.predict(Xte);

        // 6. Evaluate
        std::cout << "\n" << classification_report(s.test.y, y_pred) << "\n";

        // 7. Save predictions
        save_predictions(out_path, s.test.y, y_pred, scores);
        std::cout << "Predictions saved to " << out_path << "\n";

        // Extra results for the report
        LinearSVM unscaled(lambda, cfg);
        unscaled.fit(s.train.X, s.train.y);
        std::cout << std::fixed << std::setprecision(4)
                  << "\nTest accuracy WITH scaling   : "
                  << accuracy(s.test.y, y_pred)
                  << "\nTest accuracy WITHOUT scaling: "
                  << accuracy(s.test.y, unscaled.predict(s.test.X))
                  << "\n5-fold CV accuracy           : "
                  << cross_val_accuracy(data, 5, lambda, cfg, seed) << "\n";
    } catch (const SvmError& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}