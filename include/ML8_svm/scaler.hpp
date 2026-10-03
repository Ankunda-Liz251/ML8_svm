#ifndef ML8_SCALER_HPP
#define ML8_SCALER_HPP

#include <vector>
#include <stdexcept>

namespace ml8 {

class StandardScaler {
private:
    std::vector<double> means_;
    std::vector<double> std_devs_;
    bool is_fitted_ = false;

public:
    StandardScaler() = default;

    void fit(const std::vector<std::vector<double>>& X);
    std::vector<std::vector<double>> transform(const std::vector<std::vector<double>>& X) const;
    std::vector<std::vector<double>> fit_transform(const std::vector<std::vector<double>>& X);
    std::vector<std::vector<double>> inverse_transform(const std::vector<std::vector<double>>& X) const;

    const std::vector<double>& get_means() const { return means_; }
    const std::vector<double>& get_std_devs() const { return std_devs_; }
    bool is_fitted() const { return is_fitted_; }
};

} // namespace ml8

#endif // ML8_SCALER_HPP