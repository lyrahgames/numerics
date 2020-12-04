#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

using namespace std;

// In all cases, we use a C API which indeed fastens development in numerical
// mathematics. But we have added C++ features to further optimize the usage.

namespace jacobi {
template <typename Real>
void solve(size_t n, Real* matrix, Real* x, Real* b) {
  using real = Real;
  constexpr size_t max_it = 100;
  constexpr real min_res = 1e-8f;

  vector<real> buffer(n);
  real res = INFINITY;
  size_t k = 0;

  real* old_x = x;
  real* new_x = buffer.data();

  for (; (k < max_it) && (res > min_res); ++k) {
    res = 0;
    // For parallelization, use the biggest loop which is independent.
    // Hence, overhead of thread management may be neglected.
#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
      real tmp = 0;
      for (size_t j = 0; j < i; ++j)  //
        tmp += matrix[n * i + j] * old_x[j];
      for (size_t j = i + 1; j < n; ++j)  //
        tmp += matrix[n * i + j] * old_x[j];
      new_x[i] = (b[i] - tmp) / matrix[n * i + i];
      res += (new_x[i] - old_x[i]) * (new_x[i] - old_x[i]);
    }
    swap(old_x, new_x);
  }

  if (k & 0b01) {
    // This will not be parallelized.
    // Loop too short. Overhead of threads too big.
    for (size_t i = 0; i < n; ++i)  //
      x[i] = buffer[i];
  }
}
}  // namespace jacobi

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

// The mean error involves an extra matrix-vector multiplication.
// Hence, we have not used it inside the solvers.
template <typename Real>
auto mean_error(size_t n, Real* matrix, Real* x, Real* b) {
  Real error = 0;
#pragma omp parallel for
  for (size_t i = 0; i < n; ++i) {
    Real tmp = 0;
    for (size_t j = 0; j < n; ++j)  //
      tmp += matrix[n * i + j] * x[j];
    error += (tmp - b[i]) * (tmp - b[i]);
  }
  return sqrt(error / n);
}

int main(int argc, char** argv) {
  size_t n = 1 << 10;
  if (argc == 2) {
    stringstream input{argv[1]};
    input >> n;
  }

  // Initialize matrix and right-hand side.
  vector<float> matrix(n * n);
  vector<float> b(n, 1);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < i; ++j)
      matrix[n * i + j] = 1.0f / ((i - j) * (i - j));
    matrix[n * i + i] = 4;
    for (size_t j = i + 1; j < n; ++j)
      matrix[n * i + j] = 1.0f / ((i - j) * (i - j));
  }

  {
    vector<float> x(n, 0);
    const auto start = chrono::high_resolution_clock::now();
    jacobi::solve(n, matrix.data(), x.data(), b.data());
    const auto end = chrono::high_resolution_clock::now();
    const auto time = chrono::duration<float>(end - start).count();
    const auto error = mean_error(n, matrix.data(), x.data(), b.data());
    cout << "Jacobi Method:\n"
         << "time = " << time << " s\n"
         << "error = " << error << "\n"
         << endl;
  }
  {
    vector<float> x(n, 0);
    const auto start = chrono::high_resolution_clock::now();
    gauss_seidel::solve(n, matrix.data(), x.data(), b.data());
    const auto end = chrono::high_resolution_clock::now();
    const auto time = chrono::duration<float>(end - start).count();
    const auto error = mean_error(n, matrix.data(), x.data(), b.data());
    cout << "Gauss-Seidel Method:\n"
         << "time = " << time << " s\n"
         << "error = " << error << "\n"
         << endl;
  }

  // The comparison of both algorithms is simple when there is no
  // parallelization. The Gauss-Seidel method seems to converge faster and is
  // computed much faster. Personally, I think this has also to do with cache
  // locality. The speed-up when using Gauss-Seidel over Jacobi ranges between 3
  // and 6. But when we started to parallelize the Jacobi method, first, it was
  // still slower with a speed-up of 3 up to 6 for an element count up to 1024.
  // Going further to 16000 and more elements, the Jacobi method was faster.
  // The speed-up comparison was run on a quad-core CPU with enabled
  // hyper-threading. For large values of n, the speed-up of the parallelization
  // could indeed be shown to be 8. For smaller values, it again ranged between
  // 3 to 6. This is completely consistent with the comparison of the speed-up
  // when using the Gauss-Seidel method.
}