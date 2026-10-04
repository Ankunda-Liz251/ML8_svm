#ifndef ML8_SCALER_HPP
#define ML8_SCALER_HPP

#include <vector>

namespace ml8_svm
{

class StandardScaler
{
private:
    std::vector<double> mean;
    std::vector<double> standard_deviation;
    bool fitted;

public:
    StandardScaler();

    void fit(const std::vector<std::vector<double>>& X);

    std::vector<std::vector<double>> transform(
        const std::vector<std::vector<double>>& X) const;

    std::vector<std::vector<double>> fit_transform(
        const std::vector<std::vector<double>>& X);

    std::vector<std::vector<double>> inverse_transform(
        const std::vector<std::vector<double>>& X) const;

    const std::vector<double>& get_mean() const;

    const std::vector<double>& get_standard_deviation() const;

    bool is_fitted() const;
};
}
#endif
