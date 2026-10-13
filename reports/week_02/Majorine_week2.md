# Week 2 Progress Report

Week 2 : Implementation and unit tests.

## Completed
- Implemented `find_support_vectors` (selects samples with y_i * score(x_i) <= 1 + tol, with validation for empty input, negative tol, label values and dimension mismatches). Wrote `tests/test_support_vectors.cpp` covering a hand-checked toy set, the tolerance boundary, a non-zero bias, the no-support-vector case and invalid inputs.

## In Progress
- Registering the test with CTest and checking that every member has visible commits and PRs.

## Next Week
- Write `examples/example7_support_vectors.cpp` (support-vector analysis on the real dataset), the final-report skeleton with one heading per member, README intro/objectives/features, and the week-03 report.

## AI Use
  - Tool: Claude (Anthropic)
  - Purpose: Generated a first draft of the `find_support_vectors` implementation, its unit tests and this report template.
  - Reason: To save time on boilerplate; the code was reviewed, compiled and tested by me before committing.
