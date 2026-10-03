#include <ML8_svm/dataset.hpp>//this is the implementation of the dataset.hpp file which contains the definition of the Dataset struct and the load_csv function. 
//The Dataset struct holds the features, labels, and feature names of a dataset, while the load_csv function is a stub that throws a logic_error indicating that it is not yet implemented. 
// The implementation is placed in the ml8_svm namespace.
#include <stdexcept>

namespace ml8_svm {

Dataset load_csv([[maybe_unused]] const std::string& path,
                 [[maybe_unused]] bool has_header,
                 [[maybe_unused]] int label_col,
                 [[maybe_unused]] char delim) {
    throw std::logic_error("load_csv: not implemented yet");
}

}  // namespace ml8_svm
