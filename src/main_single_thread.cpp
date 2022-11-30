
#include <cmath>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>

#include "nmbuflowtorch/definitions.hpp"  // imports and renames important files
#include "nmbuflowtorch/layer.hpp"
#include "nmbuflowtorch/layer/dense.hpp"
#include "nmbuflowtorch/layer/sigmoid.hpp"
#include "nmbuflowtorch/loss.hpp"
#include "nmbuflowtorch/loss/cross_entropy.hpp"
#include "nmbuflowtorch/loss/mse.hpp"
#include "nmbuflowtorch/math_m.hpp"
#include "nmbuflowtorch/network.hpp"
#include "nmbuflowtorch/optimizer.hpp"
#include "nmbuflowtorch/optimizer/sgd.hpp"

// -> is for pointer objects, while . is for value objects
using namespace std;

int main(int argc, char** argv)
{
  cout << "Running single threaded program" << endl;
  // Run main program

  // Sammenligner med utregninger fra https://theneuralblog.com/forward-pass-backpropagation-example/
  int input_size = 2;
  int n_classes = 1;

  // Create network
  nmbuflowtorch::Network net;
  // define loss
  // nmbuflowtorch::Loss* loss = new nmbuflowtorch::loss::CrossEntropy();

  nmbuflowtorch::optimizer::SGD* opt = new nmbuflowtorch::optimizer::SGD(0.01);

  nmbuflowtorch::Loss* loss = new nmbuflowtorch::loss::MSE();

  net.add_loss(loss);
  net.add_optimizer(opt);

  // XOR eksempler
  Matrix X = Matrix(100, 2);
  X << 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1,
      1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1,
      1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1,
      1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1,
      1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1,
      1;

  Matrix y = Matrix(100, 1);
  y << 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
      0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
      0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0;

  // Create layers
  nmbuflowtorch::layer::Dense* dense1 = new nmbuflowtorch::layer::Dense(input_size, 1000);
  nmbuflowtorch::layer::Sigmoid* sigmoid1 = new nmbuflowtorch::layer::Sigmoid();
  nmbuflowtorch::layer::Dense* dense2 = new nmbuflowtorch::layer::Dense(dense1->output_dim(), 1000);
  nmbuflowtorch::layer::Sigmoid* sigmoid2 = new nmbuflowtorch::layer::Sigmoid();
  nmbuflowtorch::layer::Dense* dense3 = new nmbuflowtorch::layer::Dense(dense2->output_dim(), 1);
  nmbuflowtorch::layer::Sigmoid* sigmoid3 = new nmbuflowtorch::layer::Sigmoid();

  net.add_layer(dense1);
  net.add_layer(sigmoid1);
  net.add_layer(dense2);
  net.add_layer(sigmoid2);
  net.add_layer(dense3);
  net.add_layer(sigmoid3);

  net.fit(X, y, 1000, 32, 0);

  // cout << net.train_batch(X, y) << endl;

  auto y_pred = net.predict(X);
  // for (auto x : y_pred) {
  //   cout << x << endl;
  // }

  vector<int> y_true_vector(y.data(), y.data() + y.rows() * y.cols());
  cout << endl;

  cout << "ACC : " << accuracy_score(y_true_vector, y_pred) << endl;

  // Cleaning up
  // delete dense1;
  // delete sigmoid1;
  // delete dense2;
  // delete sigmoid2;
  // delete loss;
  // delete opt;
  // Could  also use smart pointers instead of manual memory management (RAII)
  // shared pointers could also be used
  // std::unique_ptr<layer::dense> dense1 = std::make_unique<layer::dense>(arguments...);
}