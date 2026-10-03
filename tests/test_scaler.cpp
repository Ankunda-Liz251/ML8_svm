#include "ml8/scaler.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

void test_standard_scaler() {
    ml8::StandardScaler scaler;

    std::vector<std::vector<double>> data = {
        {1.0, 10.0},
        {3.0, 20.0},
        {5.0, 30.0},
        {7.0, 40.0}
    };

    auto scaled_data = scaler.fit_transform(data);

    assert(scaler.is_fitted() == true);
    assert(std::abs(scaled_data[0][0] + scaled_data[3][0]) < 1e-6);

    std::cout << "StandardScaler unit tests passed successfully!" << std::endl;
}

int main() {
    test_standard_scaler();
    return 0;
}