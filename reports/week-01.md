## Completed
- Declared the Dataset struct and load_csv interface (include/ML8_svm/dataset.hpp) and a stub (src/dataset.cpp).
- Added the datasets: synthetic linearly separable (200 samples) and Iris versicolor vs virginica (100 samples), with data/input/README.md.
- Added the tests/test_dataset.cpp skeleton.



## In Progress
- Planning the CSV parser (header, delimiter, label column, error cases).


## Challenges/Blockers
- My work depend on another member's types.hpp and errors.hpp; waiting for them to be merged.
- Test helper macros from one of the members (tests/test_utils.hpp) are not available yet, so the test files are skeletons.



## Next Week
- Implement load_csv and its tests; 

## AI Use
- Tool: Claude (Anthropic)
- Purpose: First draft of the header, stub and test-skeleton files for dataset and preprocessing, and the script that generated the synthetic dataset.
- Reason: To save time on boilerplate and to check the interface design. All code was compiled, read and understood by me before committing.
