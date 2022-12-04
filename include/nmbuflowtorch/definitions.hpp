#ifndef NMBUFLOWTORCH_DEFINITIONS_H_
#define NMBUFLOWTORCH_DEFINITIONS_H_

#include <unistd.h>  // Unix things for sleep function in our case

#include <Eigen/Dense>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Matrix;
typedef Eigen::Matrix<float, Eigen::Dynamic, 1> Vector;
typedef Eigen::Array<float, 1, Eigen::Dynamic> RowVector;

#define PBWIDTH              5
#define PERCENTAGEMULTIPLIER 100
#define PBSTR                '#'  //  '\xFE'  // "█"

namespace nmbuflowtorch
{
  /// @brief Prints a progress bar to the console.
  /// @param current The current progress value.
  /// @param total The total progress value.
  /// @param verbose_level The level of verbosity.
  inline void printProgress(float current, float total, int verbose_level)
  {
    // int val = (int)(percentage * PERCENTAGEMULTIPLIER);
    // int lpad = (int)(percentage * PBWIDTH);
    // int rpad = PBWIDTH - lpad;
    // printf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");

    float percent = (PERCENTAGEMULTIPLIER * (current + 1)) / total;
    percent = percent > PERCENTAGEMULTIPLIER ? PERCENTAGEMULTIPLIER : percent;  // Prevents percentage from going over 100%
    size_t lpad = (size_t)(percent / PBWIDTH);
    size_t rpad = (size_t)(PERCENTAGEMULTIPLIER / PBWIDTH - lpad);

    std::wcout << "\r"
               << "[" << std::wstring(lpad, PBSTR) << std::wstring(rpad, ' ') << "]";
    std::cout << percent << "%"
              << " [Batch " << current + 1 << " of " << total << "]";
    std::cout.flush();
  }
};  // namespace nmbuflowtorch

#endif  // NMBUFLOWTORCH_DEFINITIONS_H_