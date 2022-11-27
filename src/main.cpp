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
#include "nmbuflowtorch/network.hpp"
#include "nmbuflowtorch/optimizer.hpp"
#include "nmbuflowtorch/optimizer/sgd.hpp"

// -> is for pointer objects, while . is for value objects
using namespace std;

int main(int argc, char** argv)
{
  // Sammenligner med utregninger fra https://theneuralblog.com/forward-pass-backpropagation-example/
  int input_size = 2;
  int n_classes = 2;

  // Create network
  nmbuflowtorch::Network net;
  // define loss
  // nmbuflowtorch::Loss* loss = new nmbuflowtorch::loss::CrossEntropy();

  nmbuflowtorch::optimizer::SGD* opt = new nmbuflowtorch::optimizer::SGD(0.001);

  nmbuflowtorch::Loss* loss = new nmbuflowtorch::loss::MSE();

  net.add_loss(loss);
  net.add_optimizer(opt);

  Matrix X = Matrix(4, 2);
  X << 0, 0, 0, 1, 1, 0, 1, 1;

  Matrix y = Matrix(4, 2);
  y << 1, 0, 0, 1, 0, 1, 1, 0;


  // Create layers
  nmbuflowtorch::layer::Dense* dense1 = new nmbuflowtorch::layer::Dense(input_size, 2);
  nmbuflowtorch::layer::Sigmoid* sigmoid1 = new nmbuflowtorch::layer::Sigmoid();
  nmbuflowtorch::layer::Dense* dense2 = new nmbuflowtorch::layer::Dense(dense1->output_dim(), n_classes);
  nmbuflowtorch::layer::Sigmoid* sigmoid2 = new nmbuflowtorch::layer::Sigmoid();

  net.add_layer(dense1);
  net.add_layer(sigmoid1);
  net.add_layer(dense2);
  net.add_layer(sigmoid2);

  cout << X << endl;
  cout << y << endl;

  net.train_batch(X, y);

}