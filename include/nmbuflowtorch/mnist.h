#ifndef NMBUFLOWTORCH_MNIST_H_
#define NMBUFLOWTORCH_MNIST_H_

#include <fstream>
#include <iostream>
#include <string>
#include "./utils.h"

namespace nmbuflowtorch {
class MNIST {
 private:
  std::string data_dir;

 public:
  Matrix train_data;
  Matrix train_labels;
  Matrix test_data;
  Matrix test_labels;

  void read_mnist_data(std::string filename, Matrix& data);
  void read_mnist_label(std::string filename, Matrix& labels);

  explicit MNIST(std::string data_dir) : data_dir(data_dir) {}
  void read();
};
}  // namespace nmbuflowtorch
#endif  // NMBUFLOWTORCH_MNIST_H_
