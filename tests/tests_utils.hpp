#pragma once

#include <iostream>
#include <stdexcept>
#include <string>

inline void test_assert(bool condition, const std::string& message)
{
    if (!condition)
    {
        std::cerr << "TEST FAILED: " << message << std::endl;
        throw std::runtime_error(message);
    }

    std::cout << "TEST PASSED: " << message << std::endl;
}

template <typename Function>
inline void expect_throw(Function function, const std::string& message)
{
    try
    {
        function();
        test_assert(false, message);
    }
    catch (...)
    {
        test_assert(true, message);
    }
}

