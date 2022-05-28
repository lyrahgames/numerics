#include <cassert>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
//
#include <lyrahgames/gnuplot_pipe.hpp>

using namespace std;

int main() {
  using real = uint32_t;
  const size_t samples = 1 << 30;

  mt19937 rng{random_device{}()};
  uniform_int_distribution<real> distribution{};

  vector<real> buffer(samples);

  for (size_t i = 0; i < samples; ++i) buffer[i] = distribution(rng);
  real* data = buffer.data() +
               (64ull - reinterpret_cast<size_t>(&buffer[0]) & (64ull - 1ull)) /
                   sizeof(real);
  assert((reinterpret_cast<size_t>(data) & (64ull - 1ull)) == 0ull);

  volatile real do_not_optimize = 0;

  fstream file{"data.txt", ios::out};

  // #pragma omp parallel for
  for (size_t stride = 1; stride <= 16; stride <= 1) {
    for (size_t block = (1ull << 10u); block <= (1ull << 25u); block <<= 1) {
      double time;
      real result = 0;
      {
        const size_t last = block * stride;
        const size_t count = samples / block;

        const auto start = chrono::high_resolution_clock::now();

        for (size_t b = 0; b < count; ++b) {
          // const auto offset = b * block;
          for (size_t i = 0; i < last; i += stride) result += buffer[i];
        }

        const auto end = chrono::high_resolution_clock::now();
        time = chrono::duration<double>(end - start).count();
      }
      do_not_optimize = result;
      cout << "result = " << result << '\n';

      const auto block_size = block * sizeof(real) / real(1024);
      const auto memory = samples * sizeof(real) / real(1024 * 1024);
      const auto bandwidth = memory / time;

      cout << setw(25) << "stride = " << setw(15) << stride * sizeof(real)
           << " B\n"
           << setw(25) << "block size = " << setw(15) << block_size << " KiB\n"
           << setw(25) << "memory = " << setw(15) << memory << " MiB\n"
           << setw(25) << "time = " << setw(15) << time << " s\n"
           << setw(25) << "bandwidth = " << setw(15) << bandwidth << " MiB/s\n"
           << endl;

      file << stride << ' ' << block_size << ' ' << bandwidth << '\n';
    }
    file << '\n';
  }
  file << flush;

  lyrahgames::gnuplot_pipe plot{};
  plot << "set logscale y 2\n"
          "splot 'data.txt' u 1:2:3 w lines\n";

  // Do not quit the program to make plot interactive.
  string line;
  getline(cin, line);
}