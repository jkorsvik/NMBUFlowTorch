#ifndef NMBUFLOWTORCH_AUTOENCODER_H_
#define NMBUFLOWTORCH_AUTOENCODER_H_

#include <omp.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>

#include "./csv_parser/csv.hpp"
#include "./definitions.hpp"  // imports and renames important files
#include "./layer.hpp"
#include "./layer/dense.hpp"
#include "./layer/sigmoid.hpp"
#include "./loss.hpp"
#include "./loss/cross_entropy.hpp"
#include "./loss/mse.hpp"
#include "./math_m.hpp"
#include "./network.hpp"
#include "./optimizer.hpp"
//#include "./optimizer/adam.hpp"
#include "./optimizer/nadam.hpp"
#include "./optimizer/sgd.hpp"

// -> is for pointer objects, while . is for value objects
using namespace std;
// CSV parser
using namespace csv;

namespace nmbuflowtorch
{
  int autoencoder_train(int argc, char** argv);
}  // namespace nmbuflowtorch
#endif  // NMBUFLOWTORCH_AUTOENCODER_H_
