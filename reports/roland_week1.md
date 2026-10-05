# ROLAND — Week 1 contribution: Hinge loss

## Completed
- Declared `hinge_loss` in `include/ML8_svm/hinge_loss.hpp`.
- Declared `hinge_subgradient` in `include/ML8_svm/hinge_loss.hpp`.
- Added the Week 1 stub implementation in `src/hinge_loss.cpp`.
- Added the Week 1 test skeleton in `tests/test_hinge_loss.cpp`.
- Recorded the mathematical derivation of the hinge-loss subgradient.
- Kept the hinge-loss module separate from L2 regularization, as required by the project formulation.

## Mathematical derivation
For one training sample, define

`z_i = y_i (w · x_i + b)`.

The hinge-loss term is

`L_i = max(0, 1 - z_i)`.

Therefore:
- If `z_i < 1`, the loss is active and its contribution to the gradient is
  `-y_i x_i` with respect to `w`, and `-y_i` with respect to `b`.
- If `z_i > 1`, the loss is zero and contributes zero gradient.
- At `z_i = 1`, the hinge function has a kink. The Week 1 design follows the project specification and uses the zero contribution at the boundary.

For a dataset of `n` samples, the subgradient contribution is:

`gw += -y_i x_i / n`

`gb += -y_i / n`

for samples satisfying `y_i (w · x_i + b) < 1`.

The regularization term is handled separately by M6.

## Week 1 status
The public interface and compilation skeleton are prepared. Full numerical-gradient validation and production implementation are planned for Week 2.

## AI use
AI  'CHAT GPT' assistance was used to help structure the hinge-loss interface, derive/check the subgradient explanation, and prepare the initial code skeleton. 
 