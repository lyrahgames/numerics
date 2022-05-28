#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <ranges>
#include <vector>
//

using namespace std;

struct sparse_matrix {
  struct triplet {
    float value;
    size_t row_index;
    size_t col_index;
  };

  sparse_matrix() = default;
  sparse_matrix(vector<triplet>& triplets) {
    ranges::sort(triplets, [](auto x, auto y) {
      return (x.row_index < y.row_index) ||
             ((x.row_index == y.row_index) && (x.col_index < y.col_index));
    });
    // Assume no doubled entries.
    size_t max_col = 0;
    size_t row = 1;
    rows.push_back(0);
    rows.push_back(0);
    for (auto t : triplets) {
      for (; row < t.row_index + 1; ++row)  //
        rows.push_back(rows[row]);
      data.push_back(t.value);
      cols.push_back(t.col_index);
      max_col = max(max_col, t.col_index);
      ++rows[row];
    }
    col = max_col + 1;
  }

  size_t col;
  vector<float> data{};
  vector<size_t> rows{};
  vector<size_t> cols{};
};

vector<float> operator*(const sparse_matrix& m, const vector<float>& v) {
  vector<float> result(v.size(), 0);
  for (size_t row = 0; row < m.rows.size() - 1; ++row) {
    for (size_t i = m.rows[row]; i < m.rows[row + 1]; ++i) {
      const auto col = m.cols[i];
      result[row] += m.data[i] * v[col];
    }
  }
  return result;
}

void normalize(vector<float>& v) {
  float norm = 0;
  for (auto x : v) norm += x * x;
  const auto inv_norm = 1.0f / sqrt(norm);
  for (auto& x : v) x *= inv_norm;
}

float error(const vector<float>& v, const vector<float>& w) {
  float result = 0;
  for (size_t i = 0; i < v.size(); ++i) result += (v[i] - w[i]) * (v[i] - w[i]);
  return result;
}

float power_iteration_eigenvalue(const sparse_matrix& m, vector<float>& x) {
  constexpr size_t max_it = 100;
  constexpr float precision = 1e-6f;

  float r = INFINITY;
  for (size_t it = 0; it < max_it && (r > precision); ++it) {
    auto y = m * x;
    normalize(y);
    r = error(y, x);
    x = y;
  }

  float result = 0;
  auto y = m * x;
  for (size_t i = 0; i < x.size(); ++i) result += x[i] * y[i];
  return result;
}

ostream& operator<<(ostream& os, const sparse_matrix& m) {
  for (size_t row = 0; row < m.rows.size() - 1; ++row) {
    size_t col = 0;
    for (size_t c = m.rows[row]; c < m.rows[row + 1]; ++c, ++col) {
      for (; col != m.cols[c]; ++col)
        os << setw(15) << "0"
           << "\t";
      os << setw(15) << m.data[c] << "\t";
    }
    for (; col < m.col; ++col)
      os << setw(15) << "0"
         << "\t";
    os << "\n";
  }

  // os << m.col << '\n';

  // for (auto x : m.data) os << x << '\t';
  // os << '\n';

  // for (auto x : m.rows) os << x << '\t';
  // os << '\n';

  // for (auto x : m.cols) os << x << '\t';
  // os << '\n';

  return os;
}

float inverse_iteration_eigenvalue(const sparse_matrix& m, vector<float>& x) {
  constexpr size_t max_it = 100;
  constexpr float precision = 1e-6f;

  vector<float> y(x.size(), 0);
  vector<float> d1(x.size());
  sparse_matrix lu;

  lu.data.resize(m.data.size() - x.size());
  lu.rows.resize(m.rows.size());
  lu.cols.resize(lu.data.size());
  lu.col = m.col;
  lu.rows[0] = 0;
  size_t e = 0;
  for (size_t row = 0; row < m.rows.size() - 1; ++row) {
    for (size_t i = m.rows[row]; i < m.rows[row + 1]; ++i) {
      const auto col = m.cols[i];
      if (col == row) {
        d1[row] = 1.0f / m.data[i];
        continue;
      }
      lu.data[e] = m.data[i];
      lu.cols[e] = m.cols[i];
      ++e;
    }
    lu.rows[row + 1] = e;
  }
  cout << '\n' << lu << '\n' << '\n' << flush;
  for (auto x : d1) cout << setw(15) << x << '\t';
  cout << '\n' << '\n';

  float r = INFINITY;
  for (size_t it = 0; it < max_it && (r > precision); ++it) {
    // solve(m, y, x);
    float res = INFINITY;
    size_t k = 0;
    for (; (k < max_it) && (res > precision); ++k) {
      res = 0;
      for (size_t i = 0; i < n; ++i) {
        float tmp = 0;
        for (size_t j = lu.rows[i]; j < lu.rows[i + 1]; ++j)
          tmp += lu.data[j] * x[lu.cols[j]];
        tmp = (b[i] - tmp) * d1[i];
        res += (tmp - x[i]) * (tmp - x[i]);
        x[i] = tmp;
      }
    }
    normalize(y);
    r = error(y, x);
    x = y;
  }

  float result = 0;
  y = m * x;
  for (size_t i = 0; i < x.size(); ++i) result += x[i] * y[i];
  return result;
}

namespace gauss_seidel {
template <typename Real>
void solve(size_t n, Real* matrix, Real* x, Real* b) {
  using real = Real;
  constexpr size_t max_it = 100;
  constexpr real min_res = 1e-8f;

  real res = INFINITY;
  size_t k = 0;
  for (; (k < max_it) && (res > min_res); ++k) {
    res = 0;
    for (size_t i = 0; i < n; ++i) {
      real tmp = 0;
      for (size_t j = 0; j < i; ++j)  //
        tmp += matrix[n * i + j] * x[j];
      for (size_t j = i + 1; j < n; ++j)  //
        tmp += matrix[n * i + j] * x[j];
      tmp = (b[i] - tmp) / matrix[n * i + i];
      res += (tmp - x[i]) * (tmp - x[i]);
      x[i] = tmp;
    }
  }
}
}  // namespace gauss_seidel

int main() {
  const size_t n = 100;
  const float h = 10.0f / (n - 1);

  vector<sparse_matrix::triplet> data{{-4, 0, 0}, {2, 1, 1}, {3, 2, 2}};
  sparse_matrix m{data};
  vector<float> v{1, 2, 3};

  for (auto x : v) cout << setw(15) << x << '\t';
  cout << '\n' << '\n';

  cout << m << '\n';

  auto w = m * v;

  for (auto x : w) cout << setw(15) << x << '\t';
  cout << '\n' << '\n';

  auto l = inverse_iteration_eigenvalue(m, v);
  for (auto x : v) cout << setw(15) << x << '\t';
  cout << '\n' << '\n';
  cout << "l = " << l << '\n';
}