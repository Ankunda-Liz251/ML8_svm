# Week 2 Report: Liz (M9), Decision Function and Binary Classification

## Completed
- Implemented `decision.cpp`: `decision_function` for one sample and for a matrix (both use `Hyperplane::score`, so the formula is not duplicated), and `to_label` (score of 0 or more gives +1, otherwise -1).
- Added input checks: wrong number of features throws `DimensionMismatch`, and a NaN score throws `InvalidArgument`.
- Implemented `linear_svm.cpp`: constructor checks (negative lambda, learning rate of 0 or less, zero epochs), `fit` wired to `optimize`, `decision_function`, `predict`, and the accessors `hyperplane()`, `history()` and `support_vector_indices()`.
- `fit` validates the data first (not empty, equal row lengths, no NaN, labels only -1 or +1, both classes present) and works on a local copy, so a failed `fit` leaves the old model unchanged.
- Wrote full unit tests in `test_decision.cpp` and `test_linear_svm.cpp`. Both pass: a toy dataset is classified with 100% accuracy, and all error cases behave as expected.

## In Progress
- Re-running my tests against the real `optimizer.cpp` (M7), `hinge_loss.cpp` (M5) and `hyperplane.cpp` (M4) as soon as they are merged. So far I tested with simple temporary stand-ins.
- Syncing with M7 and M10 at the end of the week, as the plan asks (M7 feeds M9, and M9 feeds M10).

## Challenges / Blockers
- The real optimizer was not ready, so I wrote a small stand-in with plain gradient descent to test `fit`. My results could differ slightly with M7's version.
- The contract does not say which `batch_size` values are valid, so I left it unchecked in the constructor. This still needs agreeing with M7.
- With a very small lambda the stand-in found no support vectors on separable data. This is expected, since all points end up beyond the margin.
- Test macro names depend on M3's `test_utils.hpp`. I may need to rename them if they differ.

## Source of Data
- The tests use a small hand-made toy dataset: four points near (2, 2) labelled +1 and four near (-2, -2) labelled -1. A straight line can separate it perfectly, and the expected results can be checked by hand.
- The decision tests use a fixed line (w = (2, -1), b = 0.5) with scores worked out by hand, for example the point (1, 1) scores 1.5.
- Real data from M1's `data/input/` is planned for Week 3 integration.

## Next Week
- Add edge-case tests and finish input validation and accessors.
- Write `examples/example4_train_predict.cpp`.
- Draft the usage snippets for the README.

## AI Use
- Tool: Claude (Anthropic). Purpose: help draft `decision.cpp`, `linear_svm.cpp` and the two test files. Reason: to speed up the first version. I reviewed and ran the tests and I can explain how the code works.
- Edit this line so it is accurate for what you actually did, because undeclared use is penalised.