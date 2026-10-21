// example7_support_vectors.cpp
// Trains a linear SVM on a two-class CSV file and lists its support vectors.
//
// Usage: example7_support_vectors [path/to/data.csv]
// Default path: data/input/linear_separable.csv (label in the last column, with header).

#include <iostream>
#include <string>

#include <ML8_svm/ML8_svm.hpp>

using namespace ml8_svm;

int main(int argc, char** argv) {
    const std::string path = (argc > 1) ? argv[1] : "data/input/linear_separable.csv";

    try {
        // 1. Load and preprocess.
        Dataset data = load_csv(path, /*has_header=*/true, /*label_col=*/-1, ',');
        encode_binary_labels(data);
        validate(data);
        Split split = train_test_split(data, /*test_ratio=*/0.2, /*seed=*/42);

        // 2. Scale (fit on training data only).
        StandardScaler scaler;
        Matrix X_train = scaler.fit_transform(split.train.X);
        Matrix X_test = scaler.transform(split.test.X);

        // 3. Train.
        OptimizerConfig cfg;
        cfg.learning_rate = 0.1;
        cfg.epochs = 500;
        cfg.batch_size = 16;
        cfg.tol = 1e-6;
        cfg.decay = true;
        cfg.seed = 42;
        const double lambda = 0.01;

        LinearSVM svm(lambda, cfg);
        svm.fit(X_train, split.train.y);

        // 4. Support-vector analysis on the training set.
        const Hyperplane& h = svm.hyperplane();
        const std::vector<size_t> sv = find_support_vectors(h, X_train, split.train.y);

        std::cout << "Dataset: " << path << "\n";
        std::cout << "Training samples: " << X_train.size() << "\n";
        std::cout << "Margin width (2/||w||): " << h.margin_width() << "\n";
        std::cout << "Support vectors: " << sv.size() << " of " << X_train.size() << " ("
                  << (100.0 * sv.size() / X_train.size()) << "%)\n\n";

        std::cout << "index  label  y*score  features (scaled)\n";
        for (size_t i : sv) {
            std::cout << i << "      " << split.train.y[i] << "     "
                      << split.train.y[i] * h.score(X_train[i]) << "   ";
            for (double v : X_train[i]) std::cout << v << " ";
            std::cout << "\n";
        }

        // 5. Cross-check against LinearSVM's own accessor.
        std::cout << "\nLinearSVM::support_vector_indices() agrees: "
                  << (svm.support_vector_indices() == sv ? "yes" : "NO") << "\n";

        // 6. Test accuracy for context.
        std::cout << "Test accuracy: " << accuracy(split.test.y, svm.predict(X_test)) << "\n";
    } catch (const SvmError& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}