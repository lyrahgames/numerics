#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
//
#include <lyrahgames/gnuplot_pipe.hpp>

using namespace std;

using element_type = float;
constexpr size_t buffer_size = 1ull << 33u;  // 8 GiB
constexpr size_t element_count = buffer_size / sizeof(element_type);

int main() {
  // Start up random number generation.
  mt19937 rng{random_device{}()};
  uniform_real_distribution<element_type> distribution;

  // Create and fill buffer with random elements to warm up caches and
  // prevent compiler optimization from falsifying measurement.
  vector<element_type> buffer(element_count);
  for (auto& x : buffer) x = rng();

  // Use output file for logging measurements to be able to plot them.
  fstream file("data.txt", ios::out);

  // Iterate over measurements.
  for (size_t count = (1ull << 15u); count < (1ull << 26u); count <<= 1) {
    for (size_t stride = 1; stride <= 32; ++stride) {
      element_type result = 1;
      // Do the actual measurement of iterating over the buffer and reading.
      const auto start = chrono::high_resolution_clock::now();
      {
        for (size_t offset = 0; offset < stride; ++offset)
          // size_t offset = 0;
          for (size_t i = offset; i < count * stride; i += stride)
            result += buffer[i];
      }
      const auto end = chrono::high_resolution_clock::now();
      const auto time = chrono::duration<double>(end - start).count();
      // Output the result to prevent compiler optimization from falsifying
      // results.
      cout << result << '\n';

      // Compute results.
      const auto memory =
          stride * count * sizeof(element_type) / (1024.0 * 1024.0);  // MiB
      const auto bandwidth = memory / time;

      // Paste data into plotting file.
      file << stride << ' ' << count * sizeof(element_type) / 1024.0 << ' '
           << bandwidth << '\n';
      cout << stride << ' ' << count << ' ' << bandwidth << '\n';
    }
    file << '\n';
  }
  file << flush;

  lyrahgames::gnuplot_pipe plot{};
  plot << "set logscale y\n"
       << "set hidden3d\n"
       << "splot 'data.txt' u 1:2:3 w l\n";

  // Do not quit the program to make plot interactive.
  string line;
  getline(cin, line);
}