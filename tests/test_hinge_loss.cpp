#include <ML8_svm/hinge_loss.hpp>


int main() {
    
    // Planned tests:
    // 1. Correct mean hinge-loss value on a hand-computed toy dataset.
    // 2. Zero loss when every sample satisfies y_i * score(x_i) >= 1.
    // 3. Positive loss for samples inside the margin / misclassified samples.
    // 4. Correct subgradient for samples with y_i * score(x_i) < 1.
    // 5. Correct handling at the kink y_i * score(x_i) == 1.
    // 6. Numerical-gradient comparison for the loss (Week 2).
    return 0;
}
 