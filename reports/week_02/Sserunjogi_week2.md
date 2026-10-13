### Model evaluation

**Completed**
- Implemented src/metrics.cpp: confusion_matrix, accuracy, precision, recall,
  f1_score and classification_report, with input validation
  (DimensionMismatch on size mismatch, InvalidArgument on empty input or labels
  outside {-1, +1}).
- Applied the zero-division convention: precision, recall and F1 return 0.0
  when their denominator is 0.
- Wrote tests/test_metrics.cpp with hand-computed expected values, including
  perfect and all-wrong predictions, zero-division cases and invalid input.
- All metrics tests pass under ctest.

**Next Week**
- Start the end-to-end pipeline test and example (load, preprocess, scale,
  fit, predict, evaluate), save predictions to data/output/predictions.csv.
- Optional k-fold cross-validation.

**AI Use**
- Claude (Anthropic) was used to draft
  src/metrics.cpp and tests/test_metrics.cpp. Purpose: speed up writing the
  implementation and test cases. I compiled and ran the tests, checked every
  expected value by hand and can explain each function.
