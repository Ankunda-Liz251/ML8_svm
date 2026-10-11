# Week 1 Report: Liz (M9), Decision Function and Binary Classification

## Completed
- Declared `decision function` (one sample and many samples) and `to label` in `decision.hpp`.
- Declared the `LinearSVM` class in `linear_svm.hpp`, with the constructor, `fit`, `predict`, `decision function`, `hyperplane()`, `history()` and `support_vector_indices()`.
- Wrote stub versions of `decision.cpp` and `linear_svm.cpp`. The library compiles, and `predict` throws `NotFitted` as the plan requires.
- Added plain-language comments to the headers so other members can understand them.

## In Progress
- Checking my headers against the interface contract and against M4 (Hyperplane), M7 (OptimizerConfig, TrainingHistory) and M8 (find support vectors) before the header freeze on Wed 7 Oct.
- Preparing for Week 2, when the real implementation and unit tests are due.

## Challenges / Blockers
- My headers depend on M6's `types.hpp` and `errors.hpp` and M4's `hyperplane.hpp`, so I used temporary stand-ins until the real files were merged.
- IntelliSense in VS Code could not find the `ML8_svm/` includes until the `include/` folder was added to the include path.
- The contract does not say which `batch_size` values are valid. This needs agreeing with M7.

## Source of Data
- No dataset was used this week, because the work was only declarations and stubs.
- From Week 2, training and testing data comes from M1's `data/input/` folder: the synthetic linearly separable set and the small real two-class set.
- My unit tests will use a small hand-made toy dataset (two groups of points on opposite sides of a line), so the expected results can be checked by hand.

## AI Use
- Tool: Claude (Anthropic). Purpose: help draft the header declarations, the stub files and these notes. Reason: to speed up the first version. I reviewed the declarations against the interface contract and I can explain them.
- Edit this line so it is accurate for what you actually did, because undeclared use is penalised.
