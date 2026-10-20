# Machine_Learning_8_-Support-Vector-Machine-for-binary-classification-
Project work for the members of Group 16
A reusable C++ library that trains and evaluates a linear Support Vector Machine (SVM) for binary classification, written from scratch with no machine-learning dependencies. 
The project was built by a group of 10 students as a C++ group project, taking scikit-learn as inspiration for the overall workflow.

## Project Name and Description
**ML8_svm** (namespace `ml8_svm`) is a library, not a stand-alone program. It covers the full classification pipeline: load a CSV file, preprocess it, scale the features, fit a linear SVM by gradient descent on the regularised hinge loss, predict, and evaluate. Labels are encoded as -1 / +1 and the model minimises

J(w, b) = (lambda / 2) * ||w||^2 + (1 / n) * sum_i max(0, 1 - y_i (w . x_i + b))

A prediction is the sign of the decision function w . x + b, and the margin width is 2 / ||w||.

## Project Objectives
- Design and implement a clear, reusable C++ library with a public API in `include/` and implementation in `src/`.
- Implement a linear SVM from first principles, with every component in its own module.
- Handle invalid input with a consistent exception hierarchy.
- Test each module with unit tests and an end-to-end pipeline test, runnable from a clean environment through CTest.
- Collaborate with Git and GitHub using feature branches, pull requests and code review.
- Document the library, report progress weekly, and declare all AI use.

## Main Features
- CSV dataset loading with clear errors for missing files, ragged rows and non-numeric fields.
- Preprocessing: label encoding to {-1, +1}, missing-value imputation, seeded shuffling, stratified train/test split and dataset validation.
- Feature scaling with `StandardScaler` and `MinMaxScaler` (fit on training data only).
- Hyperplane geometry: score, distance to the plane, functional and geometric margins, margin width.
- Hinge loss and its subgradient, L2 regularisation and the full objective.
- Gradient-based optimisation: batch gradient descent, SGD, mini-batch, learning-rate decay, early stopping, reproducible with a seed.
- Support vector identification.
- The `LinearSVM` class with `fit`, `predict`, `decision_function`, `hyperplane()`, `history()` and `support_vector_indices()`.
- Evaluation metrics: confusion matrix, accuracy, precision, recall, F1 and a classification report.

## Project Structure

## Requirements and Dependencies

## Installation and Build Instructions

## How to Use the Library

## Examples

## How to Run the Tests

## Limitations

## Contributions by Group Members

## How We Collaborated
