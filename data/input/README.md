# Input datasets

Both files are CSV with a header row. The last column is the label.

## linear_separable.csv

- 200 samples, 2 features (`x1`, `x2`), label in {-1, +1} (100 of each class).
- Synthetic: two Gaussian clusters centred at (2.5, 2.5) for +1 and (-2.5, -2.5)
  for -1, standard deviation 0.8, rows shuffled, values rounded to 4 decimals.
- Generated with NumPy `default_rng(42)`.
- The classes are linearly separable: the line `x1 + x2 = 0` separates them, so a
  correct linear SVM should reach 100% accuracy. Used for unit tests and examples.

## iris_versicolor_virginica.csv

- 100 samples, 4 features (`sepal_length`, `sepal_width`, `petal_length`,
  `petal_width`, all in cm), label 0 = Iris versicolor, 1 = Iris virginica
  (50 of each).
- Taken from Fisher's Iris data set (1936) as distributed with scikit-learn
  (originally the UCI Machine Learning Repository), keeping only the two classes
  versicolor and virginica. Rows are in the original order (sorted by class).
- These two classes are not perfectly linearly separable, so it is a more
  realistic test for scaling, training and evaluation.

## CSV format accepted by `load_csv`

- One sample per line; fields separated by a delimiter (comma by default, any single
  character can be given). Spaces around fields are ignored and blank lines are skipped.
- An optional header line gives the column names. Windows (CRLF) line endings and a
  UTF-8 byte-order mark (added by Excel) are handled.
- All columns are numbers except the label column (the last one by default), which must
  be a whole number such as `0`, `1`, `-1` or `+1` (`1.0` and `-1.0` are accepted too).
- A field can be wrapped in double quotes, for example `"1.5"`; the quotes are removed.
- Numbers can be written like `-2.5`, `+3`, `.5` or `1e-3`.
- An empty field, `NaN`, `nan`, `NA` or `?` in a feature column means "missing value";
  it is loaded as NaN and can be filled with `impute_missing()`.
- Not supported: quoted fields that contain the delimiter, and text (non-numeric) columns.
- Errors: a missing file throws `FileError`; an empty file, a row with the wrong number of
  fields, a bad number, a bad label or a `label_col` out of range throws `InvalidArgument`
  and the message names the file and line.
