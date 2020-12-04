#include <cassert>
#include <chrono>
#include <cmath>
#include <concepts>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
//
#include <lyrahgames/gnuplot_pipe.hpp>

using namespace std;

template <floating_point real>
void solve_2d_poisson_equation_with_bounds(size_t n, real* u, real* f, real h,
                                           size_t* iterations = nullptr) {
  assert(n > 2);

  constexpr size_t max_iterations = 100'000;
  constexpr real precision = 1e-12;

  // Estimate optimal relaxation parameter by heuristic.
  const auto eigenvalue = 1 - real(4.57) / real(n * n);
  const auto tmp = eigenvalue / (1 + sqrt(1 - eigenvalue * eigenvalue));
  const auto relaxation = 1 + tmp * tmp;

  const auto h2 = h * h;
  real p = INFINITY;
  size_t k = 0;

  for (; (k < max_iterations) && (p > precision); ++k) {
    p = 0;
    for (size_t i = 1; i < n - 1; ++i) {
      for (size_t j = 1; j < n - 1; ++j) {
        const size_t index = n * i + j;
        real tmp = u[index - 1] + u[index + 1] + u[index - n] + u[index + n];
        tmp = (1 - relaxation) * u[index] -
              relaxation * real(0.25) * (h2 * f[index] - tmp);
        p += (tmp - u[index]) * (tmp - u[index]);
        u[index] = tmp;
      }
    }
  }

  if (iterations) *iterations = k;
}

int main() {
  using real = double;

  fstream file{"step_sizes.dat", ios::out};
  for (size_t n = 10; n <= 1000; n *= 1.2) {
    const auto h = real{1.0} / (n - 1);
    vector<real> u(n * n, 0);
    vector<real> f(n * n);
    vector<real> analytic(n * n, 0);
    vector<real> error(n * n);

    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        const auto x = real(i) / (n - 1);
        const auto y = real(j) / (n - 1);
        f[n * i + j] = -(x * (1 - x) + y * (1 - y));
        analytic[n * i + j] = real(0.5) * x * (1 - x) * y * (1 - y);
      }
    }

    size_t iterations;

    const auto start = chrono::high_resolution_clock::now();

    solve_2d_poisson_equation_with_bounds(n, u.data(), f.data(), h,
                                          &iterations);

    const auto end = chrono::high_resolution_clock::now();
    const auto time = chrono::duration<real>(end - start).count();

    real max_error = 0;
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        error[n * i + j] = abs(analytic[n * i + j] - u[n * i + j]);
        max_error = max(max_error, error[n * i + j]);
      }
    }

    file << n << '\t' << h << '\t' << iterations << '\t' << max_error << '\t'
         << time << '\n';
    cout << n << '\n';
  }
  file << flush;

  {
    lyrahgames::gnuplot_pipe plot{};
    plot << "set title 'step-sizes-error'\n"  //
            "plot 'step_sizes.dat' u 2:4 w l\n";
  }
  {
    lyrahgames::gnuplot_pipe plot{};
    plot << "set title 'size-error'\n"  //
            "plot 'step_sizes.dat' u 1:4 w l\n";
  }
  {
    lyrahgames::gnuplot_pipe plot{};
    plot << "set title 'sizes-iteration'\n"  //
            "plot 'step_sizes.dat' u 1:3 w l\n";
  }
  {
    lyrahgames::gnuplot_pipe plot{};
    plot << "set logscale x\n"          //
            "set logscale y\n"          //
            "set title 'sizes-time'\n"  //
            "plot 'step_sizes.dat' u 1:5 w l\n";
  }

  const int n = 100;
  const auto h = real{1.0} / (n - 1);

  cout << "h = " << h << '\n';

  vector<real> u(n * n, 0);
  vector<real> f(n * n);
  vector<real> analytic(n * n, 0);
  vector<real> error(n * n);

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      const auto x = real(i) / (n - 1);
      const auto y = real(j) / (n - 1);
      f[n * i + j] = -(x * (1 - x) + y * (1 - y));
      analytic[n * i + j] = real(0.5) * x * (1 - x) * y * (1 - y);
    }
  }

  mt19937 rng{random_device{}()};
  uniform_real_distribution<real> distribution{0, 1};

  for (size_t i = 1; i < n - 1; ++i) {
    for (size_t j = 1; j < n - 1; ++j) {
      const size_t index = n * i + j;
      // u[index] = 1.0f;
      u[index] = distribution(rng);
    }
  }

  solve_2d_poisson_equation_with_bounds(n, u.data(), f.data(), h);

  real max_error = 0;
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      error[n * i + j] = abs(analytic[n * i + j] - u[n * i + j]);
      max_error = max(max_error, error[n * i + j]);
    }
  }

  cout << "max error = " << max_error << '\n';

  {
    fstream file{"source.dat", ios::out};
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j)  //
        file << f[n * i + j] << '\t';
      file << '\n';
    }
    file << flush;
    lyrahgames::gnuplot_pipe plot{};
    plot << "set title 'source term'\n"  //
            "plot 'source.dat' matrix with image\n";
  }

  {
    fstream file{"solution.dat", ios::out};
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j)  //
        file << u[n * i + j] << '\t';
      file << '\n';
    }
    file << flush;
    lyrahgames::gnuplot_pipe plot{};
    plot << "set title 'numerical solution'\n"  //
            "plot 'solution.dat' matrix with image\n";
  }

  {
    fstream file{"analytic.dat", ios::out};
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j)  //
        file << analytic[n * i + j] << '\t';
      file << '\n';
    }
    file << flush;
    lyrahgames::gnuplot_pipe plot{};
    plot << "set title 'analytical solution'\n"  //
            "plot 'analytic.dat' matrix with image\n";
  }

  {
    fstream file{"error.dat", ios::out};
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j)  //
        file << error[n * i + j] << '\t';
      file << '\n';
    }
    file << flush;
    lyrahgames::gnuplot_pipe plot{};
    plot << "set title 'absolute error'\n"  //
            "plot 'error.dat' matrix with image\n";
  }
}