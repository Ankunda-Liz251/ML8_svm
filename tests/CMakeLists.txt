add_executable(test_scaler
    test_scaler.cpp
)

target_link_libraries(test_scaler
    PRIVATE
        ML8_svm::ML8_svm
)

add_test(
    NAME scaler_test
    COMMAND test_scaler
)