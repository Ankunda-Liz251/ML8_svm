# Week 1 Progress Report: M4 (Hyperplane representation and margin computation)

## Completed
- Declared `Hyperplane` (w, b) and its five methods in `include/ML8_svm/hyperplane.hpp`.
- Added stub `src/hyperplane.cpp` so the library links.
- Added skeleton `tests/test_hyperplane.cpp` with a hand-computed example ready for week 2.
- Wrote the margin maths notes below.

## In Progress
- Waiting on M6's `types.hpp`, `errors.hpp`, `linalg.hpp` (dot, norm) and M3's `test_utils.hpp`.

## Challenges/Blockers
- Hyperplane depends on M6's core layer (critical path, due Tue 6 Oct).

## Next Week
- Implement all methods with validation; hand-computed tests; zero-weight edge case.

## AI Use
- Tool: (fill in, e.g. Claude)
- Purpose: Generated the initial header, stub and test skeleton for M4.
- Reason: To save time on boilerplate; reviewed and understood before committing.

---

## Margin maths notes

Hyperplane: score(x) = w.x + b; decision boundary is score(x) = 0; w is normal to it.

**Distance.** For a point x, the signed distance to the hyperplane is
distance(x) = (w.x + b) / ||w||. Positive on the +1 side, negative on the -1 side.

**Functional margin.** y * score(x). Positive iff x is correctly classified,
but scales if w and b are scaled, so it is not a true distance.

**Geometric margin.** y * score(x) / ||w||. Scale-invariant: the signed distance
to the boundary, positive when correctly classified.

**Margin width.** The margin boundaries are score(x) = +1 and score(x) = -1.
Take x1 with score +1 and x2 with score -1. Subtracting: w.(x1 - x2) = 2.
The shortest distance between the two parallel planes is along w, so
width = 2 / ||w||. Maximising the margin is therefore minimising ||w||,
which is why the objective contains (lambda/2)||w||^2.

**Edge case: w = 0.** Division by ||w|| is undefined, so `distance`,
`geometric_margin` and `margin_width` throw `InvalidArgument`. `score` and
`functional_margin` still work (they equal b and y*b).

**Worked example.** w = (3,4), b = -1, x = (1,1): score = 6, ||w|| = 5,
distance = 1.2, geometric margin (y=+1) = 1.2, margin width = 0.4.