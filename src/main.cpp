#include <cmath>
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
  int output_size = 1;

  // Create network
  nmbuflowtorch::Network net;
  // define loss
  // nmbuflowtorch::Loss* loss = new nmbuflowtorch::loss::CrossEntropy();

  nmbuflowtorch::Loss* loss = new nmbuflowtorch::loss::MSE();

  net.add_loss(loss);

  // Create layers
  nmbuflowtorch::layer::Dense* dense1 = new nmbuflowtorch::layer::Dense(input_size, 2);
  nmbuflowtorch::Layer* sigmoid1 = new nmbuflowtorch::layer::Sigmoid();
  nmbuflowtorch::layer::Dense* dense2 = new nmbuflowtorch::layer::Dense(dense1->output_dim(), output_size);
  nmbuflowtorch::Layer* sigmoid2 = new nmbuflowtorch::layer::Sigmoid();

  Matrix Wh12 = Matrix(2, 2);
  Wh12 << 0.1, 0.2, 0.3, 0.4;
  Vector b1 = Vector(2);
  b1 << 0.25, 0.25;
  dense1->set_bias(b1);
  dense1->set_weights(Wh12);
  net.add_layer(dense1);

  Matrix y = Matrix(2, 2);
  y << 0.05, 0.95, 0.05, 0.95;  //, 0.05, 0.95;

  Matrix X = Matrix(2, 2);
  X << 0.1, 0.5, 0.1, 0.5;  // 0.1, 0.5;

  net.forward(X);
  auto output = net.output();
  cout << output << endl;

  // add sigmoid layer
  net.add_layer(sigmoid1);
  net.forward(X);
  output = net.output();
  cout << output << endl;

  // add dense2 layer
  Matrix Wh34 = Matrix(2, 2);
  Wh34 << 0.5, 0.6, 0.7, 0.8;
  Vector b2 = Vector(2);
  b2 << 0.35, 0.35;
  dense2->set_bias(b2);
  dense2->set_weights(Wh34);
  net.add_layer(dense2);
  net.forward(X);
  output = net.output();
  cout << output << endl;

  // add sig2 layer
  net.add_layer(sigmoid2);
  net.forward(X);
  output = net.output();
  cout << output << endl;

  net.backward(X, y);

  // auto loss = loss_function.loss(y, output);
  // cout << loss << endl;

  // auto loss_grad = loss_function.gradient(y, output);
  // cout << loss_grad << endl;

  // auto back_sig_grad =.backward(loss_grad);
  // cout << back_sig_grad << endl;
}