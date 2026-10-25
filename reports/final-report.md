# ML8_svm Final Report

Support Vector Machine for binary classification from scratch (C++ library).

## 1. Introduction and Problem (M1) and Project Structure (M8)

<!-- M1: problem and motivation. M8: project structure. -->

## 2. Dataset Loading and Data Description (M1)

## 3. Preprocessing (M2)

## 4. Feature Scaling (M3)

### 4.1 Build and Test Setup (M3)

## 5. Hyperplane Geometry and Margin (M4)

<!-- Include a worked 2-D example and figure. -->

## 6. Hinge Loss (M5)

<!-- Include the subgradient derivation and test results. -->

## 7. Regularization (M6) Katende Joseph Travor
The regularization module adds a penalty on the size of the weight vector to the hinge loss. Without it, the weights can grow large and fit the training data too closely. The full objective is J(w, b) = (lambda / 2)‖w‖² + the mean hinge loss, where lambda ≥ 0 sets the strength of the penalty.

**Implementation.** `l2_penalty` returns (lambda / 2)‖w‖² and `l2_gradient` returns lambda · w. The optimizer adds this gradient to the hinge subgradient of w. The bias b is not penalised, because shifting the hyperplane does not make the model more complex. `objective` returns the hinge loss plus the penalty. Both L2 functions take O(d) time for d features, and `objective` takes O(n·d) because of the hinge loss over n samples. A negative or NaN lambda throws `InvalidArgument`, and `objective` also checks that X and y match and that the weights have one entry per feature.

**Tests.** `test_regularization.cpp` runs 29 checks, all passing. They use hand-computed values (w = (3, 4) with lambda = 0.1 gives a penalty of 1.25), compare `l2_gradient` with a numerical derivative, check that changing b leaves the penalty unchanged, and cover lambda = 0, w = 0, empty input and invalid input.

**Lambda sweep.** `examples/lambda_sweep.cpp` trains one model for each of 11 lambda values from 1e-4 to 10 on the two-class Iris subset (100 samples, 80 for training and 20 for testing). Features are standardised using the training set only. Every run uses the same settings: full-batch gradient descent, learning rate 0.1 with decay, 500 epochs, tolerance 1e-6 and seed 42.

| lambda | ‖w‖ | margin width | objective | train acc. | test acc. | support vectors |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0001 | 0.608 | 3.29 | 0.382 | 0.925 | 0.95 | 51 |
| 0.000316 | 0.608 | 3.29 | 0.382 | 0.925 | 0.95 | 51 |
| 0.001 | 0.608 | 3.29 | 0.382 | 0.925 | 0.95 | 51 |
| 0.00316 | 0.607 | 3.29 | 0.382 | 0.925 | 0.95 | 51 |
| 0.01 | 0.606 | 3.30 | 0.384 | 0.925 | 0.95 | 51 |
| 0.0316 | 0.603 | 3.32 | 0.389 | 0.925 | 0.95 | 51 |
| 0.1 | 0.593 | 3.37 | 0.406 | 0.925 | 0.95 | 53 |
| 0.316 | 0.562 | 3.56 | 0.454 | 0.925 | 0.95 | 56 |
| 1 | 0.483 | 4.14 | 0.568 | 0.925 | 0.90 | 63 |
| 3.16 | 0.321 | 6.23 | 0.758 | 0.925 | 0.85 | 75 |
| 10 | 0.128 | 15.64 | 0.918 | 0.900 | 0.85 | 80 |

Up to lambda ≈ 0.03 almost nothing changes: ‖w‖ stays near 0.61 and 51 samples are support vectors. From lambda = 0.1 the weights shrink steadily and the margin width (2/‖w‖) grows from 3.29 to 15.64 at lambda = 10. At that point all 80 training samples are support vectors, and test accuracy has fallen from 0.95 to 0.85. Strong regularization therefore produces underfitting.

**Limitations.** The test set has only 20 samples, so one sample changes accuracy by 5 percentage points. The sweep uses one split and one optimizer setting, so the trend is clear but small differences in accuracy are not conclusive.

### 7.1 Core Types and Error Handling (M6)

This layer gives every module the same data types, the same vector operations and the same way of reporting errors.

**Types.** `types.hpp` defines `Vector` as `std::vector<double>` and `Matrix` as a vector of vectors stored row by row, so `X[i][j]` is feature j of sample i. Labels are kept separately in a `std::vector<int>` holding -1 or +1. A plain standard-library type keeps the code easy to read, but rows are not stored contiguously in memory, which is slower than a flat array.

**Errors.** `errors.hpp` defines `SvmError`, derived from `std::runtime_error`, and four classes derived from it: `InvalidArgument`, `DimensionMismatch`, `FileError` and `NotFitted`. Each reuses its base class constructors. Exceptions were chosen over return codes so that a bad input cannot be ignored by accident. A caller can catch one specific class or catch `SvmError` to handle any library error. `InvalidArgument` and `DimensionMismatch` are used in most source files, `FileError` in the dataset loader and `NotFitted` in `LinearSVM`.

**Linear algebra.** `linalg.hpp` provides `dot`, `norm`, `add`, `sub` and `scale`, which each take O(n) time. `norm` is the square root of `dot(v, v)`. Every function that combines two vectors first calls `check_same_size`, so a size mismatch throws `DimensionMismatch` instead of reading out of bounds. The message names the failing function and both sizes, for example "dot: size mismatch (2 vs 1)". `check_X_y` validates a feature matrix and its labels: the matrix must not be empty, X and y must have the same length, and every row must have the same number of features.

**Tests.** `test_linalg.cpp` runs 35 checks, all passing. They cover normal values, empty vectors, mismatched sizes, a ragged last row, error messages and the exception hierarchy.

**Challenges.** `types.hpp` and `errors.hpp` were on the critical path, since every other module depends on them, so they were merged first and kept small. The shared `expect_throw` test helper reports a pass even when no exception is thrown, so these tests use their own check that verifies the exception type. The shape-check helpers are currently called only from the linalg and regularization code, because other modules validate their own input.


### 7.1 Core Types and Error Handling (M6)

<!-- Include lambda-sweep results. -->

## 8. Optimization (M7)

### 8.1 Design and Implementation

The M7 optimizer trains a linear Support Vector Machine using gradient-based updates. The `OptimizerConfig` structure defines the learning rate, maximum epochs, batch size, stopping tolerance, learning-rate decay option, and random seed. The `TrainingHistory` structure records the loss after each epoch, the number of epochs completed, and whether the stopping criterion was reached.

The optimizer supports batch, stochastic, and mini-batch updates through its batch-size setting. It combines the hinge-loss subgradient with the L2 regularization gradient to update the model weights and bias. Seeded random shuffling supports reproducibility. Optional learning-rate decay reduces the learning rate across epochs, while early stopping checks whether the change in loss is within the configured tolerance.

### 8.2 Testing and Results

The optimizer tests cover the update modes, reproducibility, and early-stopping behavior. The registered test suite passed both `scaler_test` and `optimizer_test`.

The loss-curve example was run on a small, separable toy dataset for 50 epochs. Loss decreased from 0.550675 at epoch 1 to 0.097213 at epoch 50, a reduction of approximately 82.3%. The run did not report convergence because it reached the maximum epoch limit. These results demonstrate decreasing training loss on this example, not proven performance on real-world data.

#### Hyperparameter Configuration

| Hyperparameter | Value |
|---|---:|
| Learning rate | 0.1 |
| Maximum epochs | 50 |
| Batch size | 2 |
| Tolerance | 0.0 |
| Learning-rate decay | Enabled |
| Random seed | 42 |
| L2 regularization (lambda) | 0.01 |

**Figure 1.** M7 optimizer loss curve over 50 epochs on the toy dataset. The HTML figure is an approximate illustration of the decreasing trend; the initial and final loss values above come from the program output.

### 8.3 Limitations and Challenges

The results depend on the dataset and selected hyperparameters. Decreasing training loss does not guarantee good generalization, and the small toy dataset is insufficient to assess practical classification performance. A tolerance of zero makes tolerance-based early stopping unlikely while loss continues to decrease. Further evaluation should use a real dataset, compare hyperparameter settings, and assess classification metrics on held-out data.

During integration, the CMake source list and a namespace mismatch in the scaler test required correction before the complete build and test suite passed.

## 9. Support Vectors (M8)

## 10. Decision Function and the LinearSVM Class (M9)

## 11. Evaluation and Results (M10)

<!-- Include results tables and the scikit-learn comparison. -->

## 12. Challenges, Lessons Learned and Limitations (all)

### 12.1 AI-Use Summary (M8)

## Appendix: Demo Script (M10)