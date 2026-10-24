#include "ML8_svm/scaler.hpp"
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace ml8_svm
{

namespace
{

void check_dataset(const std::vector<std::vector<double>>& data)
{
    if (data.empty() || data[0].empty())
        throw std::invalid_argument("Dataset cannot be empty");

    std::size_t expected_features = data[0].size();

    for (std::size_t row_index = 0; row_index < data.size(); row_index++)
    {
        if (data[row_index].size() != expected_features)
            throw std::invalid_argument("Rows must have equal features");
    }
}

void check_feature_count(const std::vector<std::vector<double>>& data,
                         std::size_t expected_features)
{
    if (data.empty())
        throw std::invalid_argument("Dataset cannot be empty");

    if (expected_features == 0)
        throw std::invalid_argument("Dataset must contain features");

    for (std::size_t row_index = 0; row_index < data.size(); row_index++)
    {
        if (data[row_index].size() != expected_features)
            throw std::invalid_argument("Incorrect number of features");
    }
}

}

StandardScaler::StandardScaler()
{
    fitted = false;
}

void StandardScaler::fit(const std::vector<std::vector<double>>& X)
{
    check_dataset(X);

    std::size_t number_of_rows = X.size();
    std::size_t number_of_features = X[0].size();
    double row_count = static_cast<double>(number_of_rows);

    mean.assign(number_of_features, 0.0);
    standard_deviation.assign(number_of_features, 0.0);

    for (std::size_t row_index = 0; row_index < number_of_rows; row_index++)
    {
        for (std::size_t feature_index = 0; feature_index < number_of_features; feature_index++)
        {
            mean[feature_index] += X[row_index][feature_index];
        }
    }

    for (std::size_t feature_index = 0; feature_index < number_of_features; feature_index++)
    {
        mean[feature_index] = mean[feature_index] / row_count;
    }

    for (std::size_t row_index = 0; row_index < number_of_rows; row_index++)
    {
        for (std::size_t feature_index = 0; feature_index < number_of_features; feature_index++)
        {
            double difference = X[row_index][feature_index] - mean[feature_index];
            standard_deviation[feature_index] += difference * difference;
        }
    }

    for (std::size_t feature_index = 0; feature_index < number_of_features; feature_index++)
    {
        double variance = standard_deviation[feature_index] / row_count;
        standard_deviation[feature_index] = std::sqrt(variance);

        if (standard_deviation[feature_index] == 0.0)
            standard_deviation[feature_index] = 1.0;
    }

    fitted = true;
}

std::vector<std::vector<double>> StandardScaler::transform(
    const std::vector<std::vector<double>>& X) const
{
    if (!fitted)
        throw std::logic_error("Fit the scaler first");

    check_feature_count(X, mean.size());

    std::vector<std::vector<double>> result = X;

    for (std::size_t row_index = 0; row_index < X.size(); row_index++)
    {
        for (std::size_t feature_index = 0; feature_index < X[row_index].size(); feature_index++)
        {
            double centered_value = X[row_index][feature_index] - mean[feature_index];
            result[row_index][feature_index] = centered_value / standard_deviation[feature_index];
        }
    }

    return result;
}

std::vector<std::vector<double>> StandardScaler::fit_transform(
    const std::vector<std::vector<double>>& X)
{
    fit(X);
    return transform(X);
}

std::vector<std::vector<double>> StandardScaler::inverse_transform(
    const std::vector<std::vector<double>>& X) const
{
    if (!fitted)
        throw std::logic_error("Fit the scaler first");

    check_feature_count(X, mean.size());

    std::vector<std::vector<double>> result = X;

    for (std::size_t row_index = 0; row_index < X.size(); row_index++)
    {
        for (std::size_t feature_index = 0; feature_index < X[row_index].size(); feature_index++)
        {
            double stretched_value = X[row_index][feature_index] * standard_deviation[feature_index];
            result[row_index][feature_index] = stretched_value + mean[feature_index];
        }
    }

    return result;
}

const std::vector<double>& StandardScaler::get_mean() const
{
    return mean;
}

const std::vector<double>& StandardScaler::get_standard_deviation() const
{
    return standard_deviation;
}

bool StandardScaler::is_fitted() const
{
    return fitted;
}

MinMaxScaler::MinMaxScaler()
{
    new_min = 0.0;
    new_max = 1.0;
    fitted = false;
}

MinMaxScaler::MinMaxScaler(double range_min, double range_max)
{
    if (range_min >= range_max)
        throw std::invalid_argument("new_min must be less than new_max");

    new_min = range_min;
    new_max = range_max;
    fitted = false;
}

void MinMaxScaler::fit(const std::vector<std::vector<double>>& X)
{
    check_dataset(X);

    std::size_t number_of_rows = X.size();
    std::size_t number_of_features = X[0].size();

    min_values.assign(number_of_features, 0.0);
    max_values.assign(number_of_features, 0.0);

    for (std::size_t feature_index = 0; feature_index < number_of_features; feature_index++)
    {
        min_values[feature_index] = X[0][feature_index];
        max_values[feature_index] = X[0][feature_index];
    }

    for (std::size_t row_index = 0; row_index < number_of_rows; row_index++)
    {
        for (std::size_t feature_index = 0; feature_index < number_of_features; feature_index++)
        {
            double value = X[row_index][feature_index];

            if (value < min_values[feature_index])
                min_values[feature_index] = value;

            if (value > max_values[feature_index])
                max_values[feature_index] = value;
        }
    }

    fitted = true;
}

std::vector<std::vector<double>> MinMaxScaler::transform(
    const std::vector<std::vector<double>>& X) const
{
    if (!fitted)
        throw std::logic_error("Fit the scaler first");

    check_feature_count(X, min_values.size());

    std::vector<std::vector<double>> result = X;
    double new_range = new_max - new_min;

    for (std::size_t row_index = 0; row_index < X.size(); row_index++)
    {
        for (std::size_t feature_index = 0; feature_index < X[row_index].size(); feature_index++)
        {
            double old_range = max_values[feature_index] - min_values[feature_index];

            if (old_range == 0.0)
            {
                result[row_index][feature_index] = new_min;
            }
            else
            {
                double fraction = (X[row_index][feature_index] - min_values[feature_index]) / old_range;
                result[row_index][feature_index] = fraction * new_range + new_min;
            }
        }
    }

    return result;
}

std::vector<std::vector<double>> MinMaxScaler::fit_transform(
    const std::vector<std::vector<double>>& X)
{
    fit(X);
    return transform(X);
}

std::vector<std::vector<double>> MinMaxScaler::inverse_transform(
    const std::vector<std::vector<double>>& X) const
{
    if (!fitted)
        throw std::logic_error("Fit the scaler first");

    check_feature_count(X, min_values.size());

    std::vector<std::vector<double>> result = X;
    double new_range = new_max - new_min;

    for (std::size_t row_index = 0; row_index < X.size(); row_index++)
    {
        for (std::size_t feature_index = 0; feature_index < X[row_index].size(); feature_index++)
        {
            double old_range = max_values[feature_index] - min_values[feature_index];

            if (old_range == 0.0)
            {
                result[row_index][feature_index] = min_values[feature_index];
            }
            else
            {
                double fraction = (X[row_index][feature_index] - new_min) / new_range;
                result[row_index][feature_index] = fraction * old_range + min_values[feature_index];
            }
        }
    }

    return result;
}

const std::vector<double>& MinMaxScaler::get_min_values() const
{
    return min_values;
}

const std::vector<double>& MinMaxScaler::get_max_values() const
{
    return max_values;
}

bool MinMaxScaler::is_fitted() const
{
    return fitted;
}

}