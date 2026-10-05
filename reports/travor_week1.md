### Travor (M6)
## **Completed:**
 - types.hpp 
 - errors.hpp merged (Vector/Matrix aliases; SvmError hierarchy: InvalidArgument, DimensionMismatch, FileError, NotFitted). 
 - linalg.hpp/.cpp with dot, norm, add, sub, scale and shape-check helpers.
 - regularization.hpp declared; regularization.cpp is a stub.




## **In Progress:**
 Week 2 : implementation of l2_penalty, l2_gradient, objective and unit tests.



## **Challenges:** 
- objective() needs Hyperplane and hinge_loss; coding against their stubs.




## **Next Week:**
 - Implementation of regularization, writing test_linalg.cpp and test_regularization.cpp.




## **AI Use:**
 Claude (Anthropic) was used to draft starter code and explanatory comments for types.hpp, errors.hpp, linalg and regularization stubs, to save time on boilerplate. I reviewed every line, compiled it with -Wall -Wextra, and can explain it.
