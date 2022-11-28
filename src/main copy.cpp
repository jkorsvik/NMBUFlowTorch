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
  int output_size = 1;

  // Create network
  nmbuflowtorch::Network net;
  // define loss
  // nmbuflowtorch::Loss* loss = new nmbuflowtorch::loss::CrossEntropy();

  nmbuflowtorch::optimizer::SGD* opt = new nmbuflowtorch::optimizer::SGD(0.001);

  nmbuflowtorch::Loss* loss = new nmbuflowtorch::loss::MSE();

  net.add_loss(loss);
  net.add_optimizer(opt);

  Matrix y = Matrix(2, 2);
  y << 0.05, 0.95, 0.05, 0.95;

  Matrix X = Matrix(2, 2);
  X << 0.1, 0.5, 0.1, 0.5;  // 0.1, 0.5;

  // Create layers
  nmbuflowtorch::layer::Dense* dense1 = new nmbuflowtorch::layer::Dense(input_size, 2);
  nmbuflowtorch::layer::Sigmoid* sigmoid1 = new nmbuflowtorch::layer::Sigmoid();
  nmbuflowtorch::layer::Dense* dense2 = new nmbuflowtorch::layer::Dense(dense1->output_dim(), output_size);
  nmbuflowtorch::layer::Sigmoid* sigmoid2 = new nmbuflowtorch::layer::Sigmoid();

  // First dense layer
  Matrix W1 = Matrix(2, 2);
  W1 << 0.1, 0.2, 0.3, 0.4;
  Vector b1 = Vector(2);
  b1 << 0.25, 0.25;
  dense1->set_bias(b1);
  dense1->set_weights(W1);
  net.add_layer(dense1);

  // Sigmoid after first dense
  net.add_layer(sigmoid1);

  // First hidden layer output

  // Second dense layer
  Matrix W2 = Matrix(2, 2);
  W2 << 0.5, 0.6, 0.7, 0.8;
  Vector b2 = Vector(2);
  b2 << 0.35, 0.35;
  dense2->set_bias(b2);
  dense2->set_weights(W2);
  net.add_layer(dense2);

  // Sigmoid after second dense
  net.add_layer(sigmoid2);

  cout << dense1->get_layer_type() << endl;

  cout << y << endl;
  auto test = colwise_max_index(y);
  cout << test << endl;
  /*
  
  
  net.forward(X);
  auto output = net.output();
  cout << output << endl;

  loss->eval(output, y);

  cout << loss->output() << endl;

  for (int i = 0; i < 100000; i++) {
    cout << net.train_batch(X, y) << endl;

  }
  */
 
  /*
  Matrix output;
  for (int i = 0; i < 50; i++)
  {
    net.forward(X);

    output = net.output();
    loss->eval(output, y);

    cout << loss->output() << endl;

    net.backward(output, y);

    net.update(*opt);
  }
 */
  /*  net.backward(X, y);


    net.update(*opt);

    net.forward(X);
    auto output_2 = net.output();
    cout << output_2 << endl;

    loss->eval(output, y);

    cout << loss->output() << endl;


    cout << "hei" << endl;*/
}