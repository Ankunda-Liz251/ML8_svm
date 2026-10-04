## Completed
- Declared the Dataset struct and load_csv interface (include/ML8_svm/dataset.hpp) and a stub (src/dataset.cpp).
- Added the datasets: synthetic linearly separable (200 samples) and Iris versicolor vs virginica (100 samples), with data/input/README.md.
- Added the tests/test_dataset.cpp skeleton.
Created scaler.hpp which declares the StandardScaler and MinMaxScaler classes and their required functions.
Created scaler.cpp which provides stub implementations for the scaler functions without implementing the actual scaling calculations.
Setting up CMakeLists.txt which configure the project library, C++17 standard, include directory, and testing.
Set up tests/CMakeLists.txt which creates and registers the scaler test with CTest.
Created test_utils.hpp which provides simple, dependency-free functions for checking test results.
Created test_scaler.cpp which creates the initial test structure for the two scaler classes.
Created .gitignore which prevents build files and unnecessary generated files from being added to Git. Created LICENSE add the project's agreed license file.

## In Progress
- Planning the CSV parser (header, delimiter, label column, error cases).


## Challenges/Blockers
- My work depend on another member's types.hpp and errors.hpp; waiting for them to be merged.
- Test helper macros from one of the members (tests/test_utils.hpp) are not available yet, so the test files are skeletons.
making the CMakeLists


## Next Week
- Implement load_csv and its tests; 
Implementing the scaler classes
## AI Use
- Tool: Claude (Anthropic)
- Purpose: First draft of the header, stub and test-skeleton files for dataset and preprocessing, and the script that generated the synthetic dataset.
- Reason: To save time on boilerplate and to check the interface design. All code was compiled, read and understood by me before committing.
 Tool: claude ai 
 purpose: to make the CMakeLists.