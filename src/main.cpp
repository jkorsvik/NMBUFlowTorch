#include <cmath>
#include <iostream>
#include <ostream>
#include <vector>

#include "nmbuflowtorch/definitions.hpp"  // imports and renames important files
#include "nmbuflowtorch/layer.hpp"
#include "nmbuflowtorch/layer/dense.hpp"
#include "nmbuflowtorch/layer/sigmoid.hpp"
#include "nmbuflowtorch/loss.hpp"
#include "nmbuflowtorch/loss/cross_entropy_loss.hpp"
#include "nmbuflowtorch/network.hpp"
#include "nmbuflowtorch/optimizer.hpp"
#include "nmbuflowtorch/optimizer/sgd.hpp"

// -> is for pointer objects, while . is for value objects
using namespace std;

int main(int argc, char** argv)
{
  // Sammenligner med utregninger fra https://theneuralblog.com/forward-pass-backpropagation-example/
  int input_size = 2;
  int output_size = 1;

  // Create network
  nmbuflowtorch::Network net;
  // define loss
  nmbuflowtorch::Loss* loss = new nmbuflowtorch::loss::CrossEntropy();
  net.add_loss(loss);

  // Create layers
  nmbuflowtorch::layer::Dense* dense1 = new nmbuflowtorch::layer::Dense(input_size, 2);
  nmbuflowtorch::Layer* sigmoid1 = new nmbuflowtorch::layer::Sigmoid();
  nmbuflowtorch::layer::Dense* dense2 = new nmbuflowtorch::layer::Dense(dense2->output_dim(), output_size);
  nmbuflowtorch::Layer* sigmoid2 = new nmbuflowtorch::layer::Sigmoid();

  Matrix W = Matrix(2, 2);
  W << 0.1, 0.2, 0.3, 0.4;

  dense1->set_weights(W);
  net.add_layer(dense1);

  Matrix y = Matrix(2, 2);
  y << 0.05, 0.95, 0.05, 0.95;

  Matrix X = Matrix(2, 2);
  X << 0.1, 0.5, 0.1, 0.5;

  net.forward(X);
  auto output = net.output();
  cout << output << endl;

  // auto output_sig = s.forward(output);
  // cout << output_sig << endl;

  // auto back_sig = s.backward(output);
  // cout << back_sig << endl;

  // auto loss = loss_function.loss(y, output);
  // cout << loss << endl;

  // auto loss_grad = loss_function.gradient(y, output);
  // cout << loss_grad << endl;

  // auto back_sig_grad =.backward(loss_grad);
  // cout << back_sig_grad << endl;
}