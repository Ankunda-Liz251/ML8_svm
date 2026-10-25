
cmake_minimum_required(VERSION 3.16)

project(ML8_svm)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_library(ML8_svm
    src/scaler.cpp
    src/linalg.cpp
    src/hyperplane.cpp
    src/hinge_loss.cpp
    src/regularization.cpp
    src/optimizer.cpp
    src/dataset.cpp
    src/decision.cpp
    src/linear_svm.cpp
    src/metrics.cpp
    src/preprocessing.cpp
    src/support_vectors.cpp
)

add_library(ML8_svm::ML8_svm ALIAS ML8_svm)

target_include_directories(ML8_svm
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

enable_testing()

add_subdirectory(tests)
add_executable(example6_loss_curve
    examples/example6_loss_curve.cpp
)

target_link_libraries(example6_loss_curve
    PRIVATE ML8_svm::ML8_svm
)
