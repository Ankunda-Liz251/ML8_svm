#include <ML8_svm/dataset.hpp>

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include <ML8_svm/errors.hpp>

namespace ml8_svm {

// ---------------------------------------------------------------------------
// Small helper functions. "static" means: only used inside this file.
// ---------------------------------------------------------------------------

// True for the characters we ignore at the start and end of a field.
static bool is_blank_char(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

// Remove blank characters from both ends of a text: "  12 " becomes "12".
static std::string trim_text(const std::string& text) {
    size_t start = 0;
    while (start < text.size() && is_blank_char(text[start])) {
        start++;
    }
    size_t end = text.size();
    while (end > start && is_blank_char(text[end - 1])) {
        end--;
    }
    return text.substr(start, end - start);
}

// Remove one pair of double quotes around a field: "1.5" (with the quotes) becomes 1.5.
// Spaces inside the quotes are removed too. A field without quotes is returned unchanged.
static std::string remove_quotes(const std::string& text) {
    if (text.size() >= 2 && text[0] == '"' && text[text.size() - 1] == '"') {
        return trim_text(text.substr(1, text.size() - 2));
    }
    return text;
}

// Cut one line into fields. "1, 2,3" with delimiter ',' gives "1", "2", "3".
// A quoted field must not contain the delimiter.
static std::vector<std::string> split_line(const std::string& line, char delim) {
    std::vector<std::string> fields;
    std::string current = "";
    for (size_t i = 0; i < line.size(); i++) {
        if (line[i] == delim) {
            fields.push_back(remove_quotes(trim_text(current)));
            current = "";
        } else {
            current += line[i];
        }
    }
    fields.push_back(remove_quotes(trim_text(current)));  // the last field has no delimiter after it
    return fields;
}

// True if the field means "value is missing": empty, NaN, nan, NA or ?
static bool is_missing(const std::string& text) {
    return text == "" || text == "NaN" || text == "nan" || text == "NA" || text == "?";
}

// If the whole text is a number, store it in "value" and return true.
static bool is_number(const std::string& text, double& value) {
    try {
        size_t used = 0;
        value = std::stod(text, &used);
        return used == text.size();  // "2x" is not a number: only 1 of 2 characters was used
    } catch (...) {
        return false;  // stod throws an error when the text is not a number at all
    }
}

// If the whole text is an integer, store it in "value" and return true.
static bool is_integer(const std::string& text, int& value) {
    try {
        size_t used = 0;
        value = std::stoi(text, &used);
        return used == text.size();  // "1.5" is not an integer
    } catch (...) {
        return false;
    }
}

// True if the text is a label: a whole number. "1", "-1" and "+1" are fine, and so are
// "1.0" and "-1.0" (some programs write labels like that). "1.5" and "nan" are not labels.
static bool is_label(const std::string& text, int& label) {
    if (is_integer(text, label)) {
        return true;
    }
    double value = 0.0;
    if (is_number(text, value) && value == std::floor(value) && value > -1000000.0 && value < 1000000.0) {
        label = static_cast<int>(value);
        return true;
    }
    return false;
}

// The start of every error message: which file and which line.
static std::string place(const std::string& path, int line_number) {
    return "load_csv: " + path + ", line " + std::to_string(line_number) + ": ";
}

// ---------------------------------------------------------------------------
// load_csv
// ---------------------------------------------------------------------------

Dataset load_csv(const std::string& path, bool has_header, int label_col, char delim) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw FileError("load_csv: cannot open file '" + path + "'");
    }

    Dataset data;
    bool first_line = true;  // true until we have read the first non-blank line
    int column_count = 0;    // number of columns in the file
    int label_index = 0;     // which column holds the label
    int line_number = 0;
    std::string line;

    while (std::getline(file, line)) {
        line_number++;

        // Files saved by Excel can start with 3 invisible bytes (a "BOM"). Remove them.
        if (line_number == 1 && line.size() >= 3 && line[0] == '\xEF' && line[1] == '\xBB' && line[2] == '\xBF') {
            line = line.substr(3);
        }

        if (trim_text(line) == "") {
            continue;  // skip blank lines
        }

        std::vector<std::string> fields = split_line(line, delim);
        int field_count = fields.size();

        if (first_line) {
            // The first line decides how many columns the file has.
            first_line = false;
            column_count = field_count;
            if (column_count < 2) {
                throw InvalidArgument(place(path, line_number) + "need at least one feature column and one label column");
            }

            // label_col = -1 means "the last column", -2 the one before it, and so on.
            if (label_col < 0) {
                label_index = column_count + label_col;
            } else {
                label_index = label_col;
            }
            if (label_index < 0 || label_index >= column_count) {
                throw InvalidArgument(place(path, line_number) + "label_col " + std::to_string(label_col)
                                      + " does not exist (the file has " + std::to_string(column_count) + " columns)");
            }

            if (has_header) {
                // This line holds the column names, not data.
                for (int c = 0; c < column_count; c++) {
                    if (c != label_index) {
                        data.feature_names.push_back(fields[c]);
                    }
                }
                continue;
            }
            // No header: make up the names x0, x1, x2, ...
            for (int k = 0; k < column_count - 1; k++) {
                data.feature_names.push_back("x" + std::to_string(k));
            }
        } else if (field_count != column_count) {
            throw InvalidArgument(place(path, line_number) + "expected " + std::to_string(column_count)
                                  + " fields but found " + std::to_string(field_count));
        }

        // This line is a data row: one label and the features.
        Vector row;
        for (int c = 0; c < column_count; c++) {
            if (c == label_index) {
                int label = 0;
                if (!is_label(fields[c], label)) {
                    throw InvalidArgument(place(path, line_number) + "label '" + fields[c] + "' is not an integer");
                }
                data.y.push_back(label);
            } else if (is_missing(fields[c])) {
                row.push_back(std::nan(""));  // NaN = "not a number"; impute_missing() fills it in later
            } else {
                double value = 0.0;
                if (!is_number(fields[c], value)) {
                    throw InvalidArgument(place(path, line_number) + "'" + fields[c] + "' in column "
                                          + std::to_string(c + 1) + " is not a number");
                }
                row.push_back(value);
            }
        }
        data.X.push_back(row);
    }

    if (first_line) {
        throw InvalidArgument("load_csv: " + path + " is empty");
    }
    if (data.X.size() == 0) {
        throw InvalidArgument("load_csv: " + path + " has a header but no data rows");
    }
    return data;
}

}  // namespace ml8_svm
