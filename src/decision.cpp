
//  decision.cpp  
//  Implementation of the functions declared in decision.hpp.

#include <ML8_svm/decision.hpp>
#include <ML8_svm/errors.hpp>

#include <cmath>    // std::isnan
#include <string>

namespace ml8_svm {

// ---- score of one sample ----------------------------------------------------
double decision_function(const Hyperplane& hyperplane, const Vector& x) {
    // Step 1: check the sample has the right number of features.
    // (We check here, instead of only inside score(), so the error message
    //  is friendly and says exactly what went wrong.)
    if (x.size() != hyperplane.w.size()) {
        throw DimensionMismatch(
            "decision_function: sample has " + std::to_string(x.size()) +
            " features but the model expects " +
            std::to_string(hyperplane.w.size()));
    }

    // Step 2: let the Hyperplane compute w . x + b.
    // We do NOT repeat the formula here, so there is one single place
    // (hyperplane.cpp) where the score is calculated.
    return hyperplane.score(x);
}

// ---- scores of many samples -------------------------------------------------
Vector decision_function(const Hyperplane& hyperplane, const Matrix& X) {
    Vector scores;
    scores.reserve(X.size());          // we already know how many scores we need

    // Go through the samples one by one and reuse the single-sample version.
    // If a row has the wrong size, that call throws and we stop right away.
    for (const Vector& row : X) {
        scores.push_back(decision_function(hyperplane, row));
    }
    return scores;
}

// ---- score -> label ---------------------------------------------------------
int to_label(double score) {
    // NaN means "not a number" (for example 0/0). Every comparison with NaN
    // is false, so without this check NaN would quietly become -1.
    if (std::isnan(score)) {
        throw InvalidArgument("to_label: score is NaN");
    }
    return (score >= 0.0) ? +1 : -1;
}

}  // namespace ml8_svm