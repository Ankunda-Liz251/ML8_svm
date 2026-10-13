# Week 2 Report: Liz (M9), Decision Function and Binary Classification

## Completed
- Implemented `decision.cpp`: `decision_function` for one sample and for a matrix (both use `Hyperplane::score`, so the formula is not duplicated), and `to_label` (score of 0 or more gives +1, otherwise -1).
- Added input checks: a wrong number of features throws `DimensionMismatch`, and a NaN score throws `InvalidArgument`.
- Implemented `linear_svm.cpp`: constructor checks (negative lambda, learning rate of 0 or less, zero epochs), `fit` wired to `optimize`, `decision_function`, `predict`, and the accessors `hyperplane()`, `history()` and `support_vector_indices()`.
- `fit` validates the data first and works on a local copy, so a failed `fit` leaves the old model unchanged.
- Wrote unit tests in `test_decision.cpp` and `test_linear_svm.cpp`. [Edit: say which tests pass, for example "decision tests pass; the training test depends on M7's optimizer".]

## In Progress
- Running the training test against M7's real optimizer once it is merged.
- Syncing with M7 and M10, since my work depends on M7 and feeds M10.

## Challenges / Blockers
- M7's real `optimizer.cpp` is not in `main` yet, so `fit` cannot be fully tested on the repo version. 
- The test helper is named `tests_utils.hpp` in the repo, but the plan calls it `test_utils.hpp`, so I had to adjust my includes.
- my tests depend on m4 (hyperplane.cpp)and she hasn't pushed the work
- The contract does not say which `batch_size` values are valid, so I left it unchecked. This still needs agreeing with M7.
- My Week 1 files were accidentally removed from `main` by a cleanup commit, and I restored them with lowercase names.

## Source of Data
- The tests use a small hand-made toy dataset: four points near (2, 2) labelled +1 and four near (-2, -2) labelled -1.
- The decision tests use a fixed line (w = (2, -1), b = 0.5) with hand-calculated scores, for example (1, 1) scores 1.5.
- Real data from M1's `data/input/` is planned for Week 3 integration.

## Next Week
- Add edge-case tests and finish input validation and accessors.
- Write `examples/example4_train_predict.cpp`.
- Draft the usage snippets for the README.

## AI Use
- Tool: Claude (Anthropic). Purpose: help draft `decision.cpp`, `linear_svm.cpp` and the two test files. Reason: to speed up the first version. I reviewed the code and ran the tests