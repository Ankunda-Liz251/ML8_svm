
#include "ML8_svm/scaler.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

void test_standard_scaler() {
   ml8_svm::StandardScaler scaler;

    std::vector<std::vector<double>> data = {
        {1.0, 10.0},
        {3.0, 20.0},
        {5.0, 30.0},
        {7.0, 40.0}
    };

    const auto scaled_data = scaler.fit_transform(data);

    assert(scaler.is_fitted());

    // The first and last standardized values should sum to zero
    // for this evenly spaced dataset.
    assert(std::abs(scaled_data[0][0] + scaled_data[3][0]) < 1e-6);

    std::cout << "StandardScaler unit test passed." << std::endl;
}

int main() {
    test_standard_scaler();
    return 0;
}
