## ROLAND — Hinge Loss (Week 2 contribution)

### Completed
- Implemented mean hinge loss: the average of max(0, 1 - y_i * score(x_i)).
- Implemented the hinge-loss subgradient for weights and bias.
- Chose the zero subgradient when a sample's margin is exactly 1.
- Added unit tests for zero loss, positive loss, mean loss, known subgradients, the margin-equals-1 case, and a numerical-gradient check.

### In progress
- Integrating the module with the shared project build and test setup.

### Challenges / blockers
- The numerical-gradient check must avoid points exactly at the hinge kink, because hinge loss is not differentiable there.

### Next week
- Add any integration or edge-case tests requested during review and fix issues found by the full project test suite.

### AI use
- OpenAI ChatGPT was used to help draft and review the implementation and test cases.


