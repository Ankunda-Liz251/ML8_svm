// =============================================================================
// errors.hpp  --  Exception hierarchy for the ML8_svm library
// Owner: Trevor (M6)
//
// WHY THIS FILE EXISTS
//   The project rule is: public functions validate their input and THROW an
//   exception instead of silently returning a wrong answer. To keep error
//   handling consistent, everybody throws the types defined here.
//
// HOW THEY ARE ORGANISED
//
//   std::runtime_error
//        |
//     SvmError                  <- catch this to catch ANY library error
//        |-- InvalidArgument    <- a value is not allowed (e.g. lambda < 0,
//        |                         empty dataset, only one class)
//        |-- DimensionMismatch  <- sizes do not fit together (e.g. dot product
//        |                         of a 3-vector and a 4-vector)
//        |-- FileError          <- a file cannot be opened or read
//        `-- NotFitted          <- model/scaler used before fit() was called
//
// HOW TO USE (examples for teammates)
//   throw ml8_svm::InvalidArgument("lambda must be >= 0");
//
//   try { svm.predict(x); }
//   catch (const ml8_svm::NotFitted& e) { std::cerr << e.what(); }
//   catch (const ml8_svm::SvmError& e)  { /* any other library error */ }
//
// Header-only: each class only needs to inherit the constructor.
// =============================================================================
#pragma once

#include <stdexcept>
#include <string>

namespace ml8_svm {

// Base class of all errors thrown by this library.
// "using std::runtime_error::runtime_error;" means: reuse the parent's
// constructors, so we can write  SvmError("message")  directly.
struct SvmError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// A function received a value that is not allowed.
struct InvalidArgument : SvmError {
    using SvmError::SvmError;
};

// Two objects that must have matching sizes do not.
struct DimensionMismatch : SvmError {
    using SvmError::SvmError;
};

// A file could not be found, opened or read.
struct FileError : SvmError {
    using SvmError::SvmError;
};

// Something that must be fitted first (LinearSVM, scalers) was used too early.
struct NotFitted : SvmError {
    using SvmError::SvmError;
};

}  // namespace ml8_svm
