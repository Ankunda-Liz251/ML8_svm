# Week 2 code , commit by commit


## M1 (branch Abiyo_Bruce)
1. src/dataset.cpp              -> dataset: implement load_csv for header, delimiter and label column
2. tests/test_dataset.cpp       -> tests: load the two bundled datasets in test_dataset
3. tests/test_dataset.cpp       -> tests: cover header, delimiter and label-column options
4. tests/test_dataset.cpp       -> tests: cover load_csv error cases
5. src/dataset.cpp              -> dataset: read empty fields and NaN/nan/NA/? as missing values
6. src/dataset.cpp              -> dataset: skip a UTF-8 byte-order mark at the start of a file
7. tests/test_dataset.cpp       -> tests: cover missing values, whitespace, CRLF line endings and BOM
8. data/input/README.md         -> data: document the CSV format accepted by load_csv

## AI Use
For AI I used Claude AI for most of the codes.
I made sure I understood most of the code before committing to my branch.  