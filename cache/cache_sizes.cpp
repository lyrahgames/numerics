#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
//
#include <omp.h>
//
#include <lyrahgames/gnuplot_pipe.hpp>

using namespace std;

struct p2lcg {
  static constexpr uint64_t multiplier = 6364136223846793005ull;
  static constexpr uint64_t increment = 1442695040888963407ULL;
  auto operator()() {
    state = multiplier * state + increment;
    return state;
  }
  uint64_t state{0xcafef00dd15ea5e5ULL};
};

int main() {
  using real = uint64_t;
  const size_t samples = 1 << 29;

  mt19937 rng{random_device{}()};
  uniform_int_distribution<real> distribution{};

  vector<real> buffer(samples);
  for (size_t i = 0; i < samples; ++i) buffer[i] = distribution(rng);

  p2lcg access{};

  volatile real do_not_optimize_bias;
  double bias_time;
  real bias_result = 0;
  {
    const auto start = chrono::high_resolution_clock::now();
#pragma omp parallel for reduction(+ : bias_result)
    for (size_t i = 0; i < samples; ++i) {
      const size_t index = access() & (samples - 1);
      bias_result += reinterpret_cast<const real&>(index);
    }
    const auto end = chrono::high_resolution_clock::now();
    bias_time = chrono::duration<double>(end - start).count();
  }
  do_not_optimize_bias = bias_result;
  cout << bias_result;

  volatile real do_not_optimize = 0;

  fstream file{"data.txt", ios::out};

  for (size_t block = 256; block <= samples; block <<= 1) {
    double time;
    real result = 0;
    {
      const auto start = chrono::high_resolution_clock::now();
#pragma omp parallel for reduction(+ : result)
      for (size_t i = 0; i < samples; ++i) {
        const size_t index = access() & (block - 1);
        result += buffer[index];
      }
      const auto end = chrono::high_resolution_clock::now();
      time = chrono::duration<double>(end - start).count();
    }
    do_not_optimize = result;
    cout << result << '\n';

    const auto block_size = block * sizeof(real) / real(1024);
    const auto read_memory_mib =
        samples * sizeof(real) / real(1024) / real(1024);
    const auto bandwidth = read_memory_mib / (time);

    cout << setw(25) << "block size = " << setw(15)
         << block * sizeof(real) / real(1024) << " KiB\n"
         << setw(25) << "read memory = " << setw(15) << read_memory_mib
         << " MiB\n"
         << setw(25) << "time = " << setw(15) << time << " s\n"
         << setw(25) << "bias = " << setw(15) << bias_time << " s\n"
         << setw(25) << "bandwidth = " << setw(15) << bandwidth << " MiB/s\n"
         << endl;

    file << block_size << ' ' << bandwidth << '\n';
  }
  file << flush;

  lyrahgames::gnuplot_pipe plot{};
  plot << "set logscale x 2\n"
          "plot 'data.txt' u 1:2 w lines\n";
}