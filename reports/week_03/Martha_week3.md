# Week 3 Progress Report: (Hyperplane representation and margin computation)

## Completed
- Zero-weight edge case finalised in `src/hyperplane.cpp`: `score` and `functional_margin` stay defined, while `distance`, `geometric_margin` and `margin_width` throw `InvalidArgument`.
- Extended `tests/test_hyperplane.cpp` with edge cases: points on the boundary, label sign flip, points on the margin edges (score = +/-1), empty weight vector, very small and very large weights.
- Wrote `examples/example3_margin.cpp`, which prints score, distance, functional margin and geometric margin for a small 2-D set, including a boundary point, a misclassified point and the zero-weight error.

## In Progress
- Hooking `example3_margin` into the root `CMakeLists.txt`.
- Header comments for `hyperplane.hpp` (week 4).

## Challenges/Blockers
- Choosing a tolerance for "zero" weights: ||w|| < 1e-12 is treated as zero, so tiny but valid weights (e.g. 1e-6) still work.
- Example needs CMake to build all examples.

## Next Week
- Document `hyperplane.hpp`, produce the margin figure and write the final-report section "Hyperplane geometry and margin" with the worked 2-D example.

## AI Use
- Tool: Claude
- Purpose: Drafted extra edge-case tests and the example program.
