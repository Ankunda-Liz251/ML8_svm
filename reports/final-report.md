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

## 7. Regularization (M6)

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