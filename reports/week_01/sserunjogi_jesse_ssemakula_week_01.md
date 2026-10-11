### M10 – Jesse (Model evaluation)

**Completed**
- Declared the evaluation API in include/ML8_svm/metrics.hpp: ConfusionMatrix
  (tp, fp, tn, fn) and confusion_matrix, accuracy, precision, recall,
  f1_score, classification_report.
- Added stub src/metrics.cpp so the module compiles and links.
- Added tests/test_metrics.cpp skeleton and the list of Week 2 test cases.
- Added data/output/.gitkeep.
- Documented conventions: +1 is the positive class; zero-division returns 0.0;
  invalid input throws DimensionMismatch / InvalidArgument.

**In Progress**
- Reviewing M1's pull request (dataset module), as per the review pairing.

**Challenges / Blockers**
- Waiting for errors.hpp (M6) and test_utils.hpp (M3) before the error-handling
  tests and shared test macros can be used.

**Next Week**
- Implement all metrics functions and the full unit tests, including
  zero-division cases.
- Sync with M7 and M9 at the end of Week 2.

**AI Use**
- <FILL IN HONESTLY>. Example: "Claude (Anthropic) was used to draft the
  header and stub layout from the group work plan. Purpose: save time on
  boilerplate. All code was reviewed, compiled and understood by me before
  committing."
  If no AI was used: "No AI tools were used this week."