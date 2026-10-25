### Travor (M6): Regularization and core layer

#### Completed
- Extended `tests/test_linalg.cpp` with edge cases:
  - Empty vectors for `dot`, `norm`, `add`, `sub` and `scale`.
  - Mismatched sizes, including one empty and one non-empty vector.
  - Invalid `X`/`y` combinations in `check_X_y`, including a ragged last row.
  - Checks that errors can be caught as `SvmError` and that messages name the failing function.
- Extended `tests/test_regularization.cpp` with:
  - Edge cases: lambda = 0, w = 0, empty w, large lambda, NaN lambda.
  - A numerical-gradient check of `l2_gradient` against `l2_penalty`.
  - A test that the bias is not penalised.
  - Invalid-input tests for `objective`.
- Wrote the lambda-sweep program `examples/lambda_sweep.cpp`: it loads the data, splits it, scales it, trains one SVM per lambda and records the results.
- Ran the sweep and saved the results to `data/output/lambda_sweep.csv`.

#### In Progress
- Reading the sweep results for the report.
- Doc comments on `types.hpp`, `errors.hpp`, `linalg.hpp` and `regularization.hpp`.

#### Challenges / Blockers
- The sweep needs working data loading, scaling, optimization and `LinearSVM` before it can run on the real dataset.

#### Next Week
- Finish the documentation comments.
- Write the "Regularization" and "Core types and error handling" report sections, including the sweep results.

#### AI Use
- **Tool:** Claude (Anthropic).
- **Purpose:** Drafted the additional edge-case tests and the lambda-sweep program.
- **Verification:** I reviewed the code, checked the expected test values by hand, and ran the tests as well as the sweep.
