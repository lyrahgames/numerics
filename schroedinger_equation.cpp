#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace std;
using real = double;

int main(int argc, char** argv) {
  if (argc != 2) {
    cout << "usage:\n" << argv[0] << " <file path>\n";
    return -1;
  }

  fstream file{argv[1], ios::out};

  const real x_min = -5;
  const real x_max = 5;
  const int m = 80;
  const int n = m - 2;
  const real h = (x_max - x_min) / (m - 1);
  const auto potential = [](real x) {
    // return (abs(x) <= 1.0f) ? 1.0f : 0.0f;
    // return x * x;
    return -10.0f;
  };
  vector<real> u(n, 1);
  vector<real> buffer(n, 0);
  const real laplacian_scale = 1.0f / (h * h);
  vector<real> v(n);
  vector<real> d(n);
  vector<real> d1(n);
  for (size_t i = 0; i < n; ++i) {
    v[i] = potential(x_min + (i + 1) * h);
    d[i] = 2.0f * laplacian_scale + v[i];
    d1[i] = 1.0f / (d[i]);
  }

  const auto solve = [&](vector<real>& x, vector<real>& b) {
    constexpr size_t max_it = 1000;
    constexpr real precision = 1e-15f;
    real r = INFINITY;
    size_t k = 0;
    for (; (k < max_it) && (r > precision); ++k) {
      r = 0;
      {
        real tmp = -laplacian_scale * x[1];
        tmp = (b[0] - tmp) * d1[0];
        r += (tmp - x[0]) * (tmp - x[0]);
        x[0] = tmp;
      }
      for (size_t i = 1; i < n - 1; ++i) {
        real tmp = -laplacian_scale * (x[i - 1] + x[i + 1]);
        tmp = (b[i] - tmp) * d1[i];
        r += (tmp - x[i]) * (tmp - x[i]);
        x[i] = tmp;
      }
      {
        real tmp = -laplacian_scale * x[n - 2];
        tmp = (b[n - 1] - tmp) * d1[n - 1];
        r += (tmp - x[n - 1]) * (tmp - x[n - 1]);
        x[n - 1] = tmp;
      }
    }
  };

  const auto power_iteration_eigenvalue = [&](vector<real>& x) {
    constexpr size_t max_it = 100;
    constexpr real precision = 1e-8f;

    real r = INFINITY;
    for (size_t it = 0; it < max_it && (r > precision); ++it) {
      real norm = 0;

      buffer[0] = d[0] * x[0] - laplacian_scale * x[0 + 1];
      norm += buffer[0] * buffer[0];
      for (size_t i = 1; i < n - 1; ++i) {
        buffer[i] = d[i] * x[i] - laplacian_scale * (x[i - 1] + x[i + 1]);
        norm += buffer[i] * buffer[i];
      }
      buffer[n - 1] = d[n - 1] * x[n - 1] - laplacian_scale * x[n - 2];
      norm += buffer[n - 1] * buffer[n - 1];

      real norm1 = 1.0f / sqrt(norm);
      for (size_t i = 0; i < n; ++i) buffer[i] *= norm1;

      r = 0;
      for (size_t i = 0; i < n; ++i)
        r += (buffer[i] - x[i]) * (buffer[i] - x[i]);

      swap(x, buffer);
    }

    real result = 0;

    buffer[0] = d[0] * x[0] - laplacian_scale * x[0 + 1];
    for (size_t i = 1; i < n - 1; ++i)
      buffer[i] = d[i] * x[i] - laplacian_scale * (x[i - 1] + x[i + 1]);
    buffer[n - 1] = d[n - 1] * x[n - 1] - laplacian_scale * x[n - 2];

    for (size_t i = 0; i < n; ++i) result += x[i] * buffer[i];

    return result;
  };

  const auto inverse_iteration_eigenvalue = [&](vector<real>& x) {
    constexpr size_t max_it = 1000;
    constexpr real precision = 1e-8f;

    real r = INFINITY;
    for (size_t it = 0; it < max_it && (r > precision); ++it) {
      solve(buffer, x);

      real norm = 0;
      for (size_t i = 0; i < n; ++i) norm += buffer[i] * buffer[i];
      real norm1 = 1.0f / sqrt(norm);
      for (size_t i = 0; i < n; ++i) buffer[i] *= norm1;

      r = 0;
      for (size_t i = 0; i < n; ++i)
        r += (buffer[i] - x[i]) * (buffer[i] - x[i]);

      swap(x, buffer);
    }

    real result = 0;

    buffer[0] = d[0] * x[0] - laplacian_scale * x[1];
    for (size_t i = 1; i < n - 1; ++i)
      buffer[i] = d[i] * x[i] - laplacian_scale * (x[i - 1] + x[i + 1]);
    buffer[n - 1] = d[n - 1] * x[n - 1] - laplacian_scale * x[n - 2];

    for (size_t i = 0; i < n; ++i) result += x[i] * buffer[i];

    return result;
  };

  const real l = inverse_iteration_eigenvalue(u);

  cout << "smallest eigenvalue = " << l << '\n';

  for (size_t i = 0; i < n; ++i)
    file << (x_min + (i + 1) * h) << '\t' << v[i] << '\t'
         << u[i] * sqrt(n) * 0.2 << '\n';
}