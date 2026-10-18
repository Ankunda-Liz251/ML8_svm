# Week 2 Report: Data Preprocessing 

## What I did

This week I wrote the full preprocessing module and its tests. It has five functions:

| Function | What it does |
|---|---|
| `encode_binary_labels` | Changes the two class labels into -1 and +1 |
| `impute_missing` | Replaces missing values (NaN) with the column mean |
| `shuffle` | Mixes the samples using a seed, so the result can be repeated |
| `validate` | Checks the dataset is not empty, rows match, and there are exactly two classes |
| `train_test_split` | Splits the data into train and test sets, keeping both classes in each |

Files: `preprocessing.hpp`, `preprocessing.cpp`, `test_preprocessing.cpp`

## Main decisions

- The smaller label becomes -1 and the larger becomes +1, so {0,1} and {3,7} both work.
- I wrote my own shuffle so the same seed gives the same order on any computer.
- The split is done per class. This makes sure the training and test sets both contain each class.
- Bad input throws an error (`InvalidArgument` or `DimensionMismatch`) instead of failing silently.

## Testing

I wrote tests for each function, including bad inputs such as a wrong number of classes, empty data, uneven rows and an invalid split ratio. Tests passed: ____ / ____

## AI Use
-i used Claude for this to rectify the sort of errors involved