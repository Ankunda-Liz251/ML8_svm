// Builds a hand-made 2-D hyperplane and prints, for a few labelled points,
// the score, signed distance, functional margin and geometric margin.

#include <ML8_svm/hyperplane.hpp>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

using namespace ml8_svm;

int main() {
    // Hyperplane x1 + x2 - 2 = 0  (w = (1,1), b = -2, ||w|| = sqrt(2)).
    Hyperplane h;
    h.w = {1.0, 1.0};
    h.b = -2.0;

    struct Sample { Vector x; int y; const char* note; };
    const std::vector<Sample> samples = {
        {{2.0, 2.0},  +1, "correct, positive side"},
        {{3.0, 1.0},  +1, "correct, positive side"},
        {{0.0, 0.0},  -1, "correct, negative side"},
        {{-1.0, 1.0}, -1, "correct, negative side"},
        {{1.0, 1.0},  +1, "on the boundary"},
        {{2.0, 1.0},  -1, "misclassified"},
    };

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Hyperplane: w = (" << h.w[0] << ", " << h.w[1]
              << "), b = " << h.b << "\n";
    std::cout << "Margin width (2/||w||): " << h.margin_width() << "\n\n";

    std::cout << std::setw(12) << "x" << std::setw(4) << "y"
              << std::setw(10) << "score" << std::setw(10) << "distance"
              << std::setw(12) << "functional" << std::setw(11) << "geometric"
              << "  note\n";

    for (const auto& s : samples) {
        std::ostringstream pt;
        pt << std::fixed << std::setprecision(1) << "(" << s.x[0] << ", " << s.x[1] << ")";
        std::cout << std::setw(12) << pt.str()
                  << std::setw(4) << s.y
                  << std::setw(10) << h.score(s.x)
                  << std::setw(10) << h.distance(s.x)
                  << std::setw(12) << h.functional_margin(s.x, s.y)
                  << std::setw(11) << h.geometric_margin(s.x, s.y)
                  << "  " << s.note << "\n";
    }

    // Edge case: a zero weight vector has no defined distance or margin width.
    Hyperplane degenerate;
    degenerate.w = {0.0, 0.0};
    degenerate.b = 0.5;
    try {
        degenerate.margin_width();
    } catch (const InvalidArgument& e) {
        std::cout << "\nZero-weight hyperplane: " << e.what() << "\n";
    }
    return 0;
}