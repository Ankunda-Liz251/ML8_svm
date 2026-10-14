#include <ML8_svm/preprocessing.hpp>
#include <ML8_svm/errors.hpp>

#include <cmath>
#include <iostream>
#include <limits>
#include <set>

using namespace ml8_svm;

// Local helpers; swap for M3's tests/test_utils.hpp macros if preferred.
static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { ++g_fail; \
    std::cerr << __FILE__ << ':' << __LINE__ << " FAILED: " #c "\n"; } } while (0)
#define CHECK_THROWS(expr, Ex) do { bool ok_ = false; \
    try { expr; } catch (const Ex&) { ok_ = true; } catch (...) {} \
    if (!ok_) { ++g_fail; std::cerr << __FILE__ << ':' << __LINE__ \
    << " FAILED: expected " #Ex "\n"; } } while (0)

// n samples; X[i] = {i}; labels a (even i) / b (odd i)
static Dataset make(size_t n, int a, int b) {
    Dataset d;
    for (size_t i = 0; i < n; ++i) {
        d.X.push_back({static_cast<double>(i)});
        d.y.push_back(i % 2 == 0 ? a : b);
    }
    return d;
}

static void test_encode() {
    Dataset d = make(4, 0, 1);
    encode_binary_labels(d);
    CHECK((d.y == std::vector<int>{-1, 1, -1, 1}));
    Dataset e = make(4, 3, 7);
    encode_binary_labels(e);
    CHECK((e.y == std::vector<int>{-1, 1, -1, 1}));
    Dataset f = make(4, -1, 1);
    encode_binary_labels(f);
    CHECK((f.y == std::vector<int>{-1, 1, -1, 1}));
    Dataset g = make(4, 0, 1);
    g.y[1] = 2;
    CHECK_THROWS(encode_binary_labels(g), InvalidArgument);
    Dataset one = make(4, 1, 1);
    CHECK_THROWS(encode_binary_labels(one), InvalidArgument);
}

static void test_impute() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    Dataset d;
    d.X = {{1.0, 10.0}, {nan, 20.0}, {3.0, nan}};
    d.y = {0, 1, 0};
    impute_missing(d);
    CHECK(std::fabs(d.X[1][0] - 2.0) < 1e-12);
    CHECK(std::fabs(d.X[2][1] - 15.0) < 1e-12);
    Dataset bad;
    bad.X = {{nan}, {nan}};
    bad.y = {0, 1};
    CHECK_THROWS(impute_missing(bad), InvalidArgument);
    Dataset ragged;
    ragged.X = {{1.0}, {1.0, 2.0}};
    ragged.y = {0, 1};
    CHECK_THROWS(impute_missing(ragged), DimensionMismatch);
}

static void test_shuffle() {
    Dataset a = make(20, 0, 1), b = make(20, 0, 1);
    shuffle(a, 42); shuffle(b, 42);
    CHECK(a.X == b.X && a.y == b.y);                 // reproducible
    std::set<int> ids;
    bool paired = true;
    for (size_t i = 0; i < a.X.size(); ++i) {
        const int id = static_cast<int>(a.X[i][0]);
        ids.insert(id);
        if (a.y[i] != id % 2) paired = false;        // X and y still match
    }
    CHECK(ids.size() == 20 && paired);
    Dataset orig = make(20, 0, 1);
    CHECK(a.X != orig.X);                            // order changed
    Dataset bad = make(3, 0, 1);
    bad.y.pop_back();
    CHECK_THROWS(shuffle(bad, 1), DimensionMismatch);
}

static void test_validate() {
    validate(make(4, 0, 1));                         // must not throw
    Dataset empty;
    CHECK_THROWS(validate(empty), InvalidArgument);
    Dataset mism = make(4, 0, 1);
    mism.y.pop_back();
    CHECK_THROWS(validate(mism), DimensionMismatch);
    CHECK_THROWS(validate(make(4, 1, 1)), InvalidArgument);
    Dataset ragged = make(4, 0, 1);
    ragged.X[2].push_back(9.0);
    CHECK_THROWS(validate(ragged), DimensionMismatch);
    Dataset nofeat = make(4, 0, 1);
    for (auto& r : nofeat.X) r.clear();
    CHECK_THROWS(validate(nofeat), InvalidArgument);
}

static void test_split() {
    Dataset d = make(10, 0, 1);                      // 5 per class
    Split s = train_test_split(d, 0.2, 7);
    CHECK(s.train.X.size() == 8 && s.test.X.size() == 2);
    CHECK(s.train.y.size() == 8 && s.test.y.size() == 2);
    int pos = 0;
    for (int v : s.test.y) pos += v;                 // stratified: one of each class
    CHECK(pos == 1);
    std::set<int> ids;                               // disjoint and complete
    for (const auto& r : s.train.X) ids.insert(static_cast<int>(r[0]));
    for (const auto& r : s.test.X) ids.insert(static_cast<int>(r[0]));
    CHECK(ids.size() == 10);
    Split t = train_test_split(d, 0.2, 7);           // same seed -> same split
    CHECK(s.train.X == t.train.X && s.test.X == t.test.X);
    CHECK_THROWS(train_test_split(d, 0.0, 1), InvalidArgument);
    CHECK_THROWS(train_test_split(d, 1.0, 1), InvalidArgument);
    CHECK_THROWS(train_test_split(make(4, 1, 1), 0.5, 1), InvalidArgument);
    Dataset tiny = make(3, 0, 1);                    // class 1 has a single sample
    CHECK_THROWS(train_test_split(tiny, 0.5, 1), InvalidArgument);
}

int main() {
    test_encode(); test_impute(); test_shuffle(); test_validate(); test_split();
    if (g_fail) { std::cerr << g_fail << " check(s) failed\n"; return 1; }
    std::cout << "test_preprocessing: all tests passed\n";
    return 0;
}