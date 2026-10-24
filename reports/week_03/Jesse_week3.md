## (Model evaluation + pipeline)

**Completed**
- Wrote tests/test_pipeline.cpp: end-to-end test (load_csv, label encoding,
  validation, stratified split, scaling fitted on train only, fit, predict,
  evaluate, save predictions). Also checks reproducibility with a fixed seed
  and that a missing file surfaces as FileError.
- Wrote examples/example5_full_pipeline.cpp: runs the full pipeline, prints
  the classification report, saves data/output/predictions.csv, and compares
  accuracy with and without scaling.
- Added a 5-fold cross-validation in the example (scaler fitted per fold).
- Generated data/output/predictions.csv.

**In Progress**
- Run on the real dataset and record results for the final report.

**Next Week**
- Results tables and scikit-learn comparison (computed offline in Python).
- README parts (examples, limitations), live-demo preparation, second
  clean-environment test on a different machine.

**AI Use**
- "Claude (Anthropic) was used to draft test_pipeline.cpp and example5_full_pipeline.cpp. Purpose: speed up writing integration code. I compiled and ran them, checked the results and
can explain every step."
