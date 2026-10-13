# Week 2 Progress Report: Hyperplane representation and margin computation

## Completed
- Implemented all `Hyperplane` methods in `src/hyperplane.cpp`: `score`, `distance`, `functional_margin`, `geometric_margin`, `margin_width`.
- Input validation: `DimensionMismatch` when x and w differ in size; `InvalidArgument` for labels outside {-1, +1} and for a zero weight vector (where a division by ||w|| is needed).
- `geometric_margin` reuses `distance`, so there is no duplicated code.
- Wrote `tests/test_hyperplane.cpp` with hand-computed expected values (w = (3,4), b = -1, ||w|| = 5): score, distance, both margins, margin width, scale invariance, wrong-size input, bad labels and the zero-weight case. All tests pass locally.

## In Progress
- Registering the test with CTest (M3's `tests/CMakeLists.txt`) and switching to `test_utils.hpp` macros once merged.
- Opening the PR for review

## Challenges/Blockers
- Deciding how to treat w = 0: `score` and `functional_margin` stay defined (they do not divide by ||w||), while `distance`, `geometric_margin` and `margin_width` throw.

## Next Week
- Zero-weight edge-case tests (extra), `examples/example3_margin.cpp` printing margins and distances, and a header comment pass.

## AI Use
- Tool: Claude AI

- Reason: To speed up writing and to cross-check hand-computed values; I verified the maths by hand and can explain each function.