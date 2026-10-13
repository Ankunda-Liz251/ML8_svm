### Trevor (M6): Regularization and core layer

#### Completed
- Implemented `l2_penalty`: returns `(lambda / 2) * ||w||^2`.
- Implemented `l2_gradient`: returns `lambda * w` (the bias is not penalised).
- Implemented `objective`: returns `hinge_loss + l2_penalty`.
- Added input validation:
  - `InvalidArgument` for a negative lambda or an empty dataset.
  - `DimensionMismatch` when `X`, `y` or the weight vector sizes do not match.
- Wrote `tests/test_linalg.cpp`, covering `dot`, `norm`, `add`, `sub`, `scale` and the shape checks.
- Wrote `tests/test_regularization.cpp` with hand-computed expected values (for example, w = (3, 4) and lambda = 0.1 gives a penalty of 1.25).

#### In Progress
- Edge-case tests for empty and mismatched inputs (Week 3).
- Planning the lambda-sweep experiment.

#### Challenges / Blockers
- `objective()` depends on M5's `hinge_loss`, so its tests cannot pass until that function is implemented.
- Test macro names depend on M3's `test_utils.hpp`; I need to confirm them with M3.

#### Next Week
- Add empty-input and mismatched-input tests.
- Run the lambda sweep and save the results to `data/output/lambda_sweep.csv`.

#### AI Use
- **Tool:** Claude (Anthropic).
- **Purpose:** Drafted the regularization implementation and the two test files.
- **Verification:** I reviewed the code, checked the expected test values by hand, and ran the tests.
