### Travor (M6): Regularization and core layer

#### Completed
- Rewrote the doc comments in `types.hpp`, `errors.hpp`, `linalg.hpp` and `regularization.hpp`. Only comments changed; the code is identical.
- Adapted `tests/test_linalg.cpp` and `tests/test_regularization.cpp` to the shared `tests_utils.hpp` helper, so they compile and pass under CTest (35 and 29 checks).
- Registered both tests in `tests/CMakeLists.txt`.
- Re-ran the lambda sweep on the Iris subset with the merged optimizer and saved the results to `data/output/lambda_sweep.csv`.
- Wrote the final-report sections "Regularization" and "Core Types and Error Handling".
- Added my contribution lines to `README.md`.

#### In Progress
- Proofreading my report sections after review.

#### Challenges / Blockers
- The tests I first wrote used a helper that was not in the repository, so they did not compile.
- The shared `expect_throw` helper reports a pass even when no exception is thrown. The tests use their own check of the exception type instead.
- The first sweep results came from a pre-merge optimizer and differed from the merged one, so the sweep was re-run after the merge.

#### Next Week
- Fix any review comments and re-run the sweep if the optimizer changes.
- Prepare my slide on the core layer and the regularization results for the presentation.

#### AI Use
- **Tool:** Claude (Anthropic).
- **Purpose:** Drafted the header comments, the adapted test files and the first version of the two report sections.
- **Verification:** I reviewed and rewrote the text in my own words, built the project, ran the tests, and checked the report numbers against `lambda_sweep.csv`.
