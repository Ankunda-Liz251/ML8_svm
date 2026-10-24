#include <ML8_svm/ML8_svm.hpp>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace ml8_svm;
namespace fs = std::filesystem;

static int g_failures = 0;
#define CHECK(cond)                                                    \
    do {                                                               \
        if (!(cond)) {                                                 \
            std::cerr << "FAILED: " #cond " (" << __FILE__ << ":"      \
                      << __LINE__ << ")\n";                            \
            ++g_failures;                                              \
        }                                                              \
    } while (0)

// Linearly separable by construction: class 1 has x1,x2 in [1,3],
// class 0 has x1,x2 in [-3,-1]. Uses raw mt19937 output so the data is
// identical on every compiler.
static void write_synthetic_csv(const std::string& path, std::size_t n) {
    std::mt19937 rng(12345);
    auto u = [&rng] {
        return static_cast<double>(rng()) / static_cast<double>(rng.max());
    };
    std::ofstream out(path);
    out << std::setprecision(10);
    out << "f1,f2,label\n";
    for (std::size_t i = 0; i < n; ++i) {
        const int label = static_cast<int>(i % 2);
        const double sign = label == 1 ? 1.0 : -1.0;
        out << sign * (1.0 + 2.0 * u()) << ',' << sign * (1.0 + 2.0 * u())
            << ',' << label << '\n';
    }
}

struct PipelineResult {
    std::size_t n_total{0}, n_train{0}, n_test{0};
    std::vector<int> y_true, y_pred;
    Vector scores;
};

static PipelineResult run_pipeline(const std::string& csv, unsigned seed) {
    Dataset data = load_csv(csv);  // header, label in last column
    encode_binary_labels(data);    // {0,1} -> {-1,+1}
    impute_missing(data);
    validate(data);

    Split s = train_test_split(data, 0.25, seed);

    StandardScaler scaler;  // fit on TRAINING data only
    const Matrix Xtr = scaler.fit_transform(s.train.X);
    const Matrix Xte = scaler.transform(s.test.X);

    OptimizerConfig cfg{};
    cfg.learning_rate = 0.1;
    cfg.epochs = 300;
    cfg.batch_size = 32;
    cfg.tol = 1e-6;
    cfg.decay = true;
    cfg.seed = seed;

    LinearSVM model(0.01, cfg);
    model.fit(Xtr, s.train.y);

    PipelineResult r;
    r.n_total = data.X.size();
    r.n_train = s.train.X.size();
    r.n_test = s.test.X.size();
    r.y_true = s.test.y;
    r.y_pred = model.predict(Xte);
    r.scores = model.decision_function(Xte);
    return r;
}

static void test_full_pipeline() {
    const std::string csv =
        (fs::temp_directory_path() / "ml8_svm_pipeline_test.csv").string();
    const std::string out_csv =
        (fs::temp_directory_path() / "ml8_svm_pipeline_preds.csv").string();
    write_synthetic_csv(csv, 200);

    const PipelineResult r = run_pipeline(csv, 42);

    // Split sizes add up and the test set is about 25%
    CHECK(r.n_total == 200);
    CHECK(r.n_train + r.n_test == r.n_total);
    CHECK(r.n_test >= 45 && r.n_test <= 55);

    // Output is well-formed
    CHECK(r.y_pred.size() == r.y_true.size());
    CHECK(r.scores.size() == r.y_true.size());
    for (int p : r.y_pred) CHECK(p == -1 || p == 1);

    // Separable data: the SVM should classify (almost) everything correctly
    CHECK(accuracy(r.y_true, r.y_pred) >= 0.95);
    CHECK(f1_score(r.y_true, r.y_pred) >= 0.95);

    // Confusion matrix is consistent with the sample count
    const ConfusionMatrix cm = confusion_matrix(r.y_true, r.y_pred);
    CHECK(cm.tp + cm.fp + cm.tn + cm.fn == r.n_test);

    // Predictions can be saved: header + one row per test sample
    {
        std::ofstream f(out_csv);
        f << "index,y_true,y_pred,score\n";
        for (std::size_t i = 0; i < r.y_true.size(); ++i)
            f << i << ',' << r.y_true[i] << ',' << r.y_pred[i] << ','
              << r.scores[i] << '\n';
    }
    std::ifstream in(out_csv);
    std::size_t lines = 0;
    std::string line;
    while (std::getline(in, line)) ++lines;
    CHECK(lines == r.n_test + 1);

    std::remove(csv.c_str());
    std::remove(out_csv.c_str());
}

// Same seed, same data -> identical predictions
static void test_pipeline_is_reproducible() {
    const std::string csv =
        (fs::temp_directory_path() / "ml8_svm_pipeline_repro.csv").string();
    write_synthetic_csv(csv, 120);

    const PipelineResult a = run_pipeline(csv, 7);
    const PipelineResult b = run_pipeline(csv, 7);
    CHECK(a.y_true == b.y_true);
    CHECK(a.y_pred == b.y_pred);

    std::remove(csv.c_str());
}

// Missing file must surface as FileError through the whole pipeline
static void test_missing_file_throws_file_error() {
    bool caught = false;
    try {
        run_pipeline("this_file_does_not_exist.csv", 1);
    } catch (const FileError&) {
        caught = true;
    } catch (...) {
    }
    CHECK(caught);
}

int main() {
    try {
        test_full_pipeline();
        test_pipeline_is_reproducible();
        test_missing_file_throws_file_error();
    } catch (const std::exception& e) {
        std::cerr << "test_pipeline: unexpected exception: " << e.what()
                  << "\n";
        return 1;
    }
    if (g_failures == 0) {
        std::cout << "test_pipeline: all checks passed\n";
        return 0;
    }
    std::cerr << "test_pipeline: " << g_failures << " check(s) FAILED\n";
    return 1;
}