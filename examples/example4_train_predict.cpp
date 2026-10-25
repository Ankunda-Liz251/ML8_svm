
//  example4_train_predict.cpp     
//
//  Shows the whole "train, then predict" story in the simplest way:
//    1. make a small two-class data set (two blobs of points)
//    2. train a LinearSVM
//    3. look at what it learned (weights, bias, margin, support vectors)
//    4. predict labels for new points
//
//  Build with CMake, then run:   ./example4_train_predict

#include <ML8_svm/linear_svm.hpp>   // in the real repo you can use <ML8_svm/ML8_svm.hpp>

#include <iostream>
#include <random>

using namespace ml8_svm;

int main() {
    // Step 1: create data
    // Class +1 sits around (2, 2); class -1 sits around (-2, -2).
    // A fixed seed (7) means the numbers are the same on every run.
    std::mt19937 random_engine(7);
    std::normal_distribution<double> noise(0.0, 0.7);   // small random wiggle

    Matrix X;
    std::vector<int> y;
    for (int i = 0; i < 20; ++i) {
        X.push_back({ 2.0 + noise(random_engine),  2.0 + noise(random_engine)});
        y.push_back(+1);
        X.push_back({-2.0 + noise(random_engine), -2.0 + noise(random_engine)});
        y.push_back(-1);
    }

    //  Step 2: create and train the model 
    double lambda = 0.3;                        // regularisation strength (try 0.01 to see fewer support vectors)
    OptimizerConfig settings;
    settings.learning_rate = 0.1;               // size of each learning step
    settings.epochs        = 500;               // maximum number of passes over the data
    settings.batch_size    = X.size();          // use all samples per step
    settings.tol           = 1e-9;              // stop early when loss barely changes
    settings.decay         = false;             // keep the learning rate constant
    settings.seed          = 42;                // for reproducible results

    LinearSVM model(lambda, settings);
    model.fit(X, y);

    // Step 3: look at what was learned 
    const Hyperplane& line = model.hyperplane();
    std::cout << "Learned line:  w = (" << line.w[0] << ", " << line.w[1]
              << "),  b = " << line.b << "\n";
    std::cout << "Margin width:  " << line.margin_width() << "\n";
    std::cout << "Training ran for " << model.history().epochs_run << " epochs, "
              << (model.history().converged ? "converged" : "did not converge") << ".\n";
    std::cout << "Support vectors: " << model.support_vector_indices().size()
              << " of " << X.size() << " training points\n\n";

    //  Step 4: predict 
    // Accuracy on the training data: count the correct predictions.
    std::vector<int> predicted = model.predict(X);
    int correct = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        if (predicted[i] == y[i]) ++correct;
    }
    std::cout << "Training accuracy: " << correct << "/" << y.size() << "\n\n";

    // Predict a few brand-new points.
    Matrix new_points = {{3.0, 1.0}, {-1.0, -3.0}, {0.2, 0.1}};
    for (const Vector& p : new_points) {
        std::cout << "Point (" << p[0] << ", " << p[1] << ")  score = "
                  << model.decision_function(p) << "  ->  class "
                  << (model.predict(p) > 0 ? "+1" : "-1") << "\n";
    }
    return 0;
}