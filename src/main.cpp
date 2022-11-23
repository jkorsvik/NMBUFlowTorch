#include <cmath>
#include <iostream>
#include <ostream>

#include "Eigen/Dense"
#include "nmbuflowtorch/layer.hpp"
#include "nmbuflowtorch/layer/dense.hpp"
#include "nmbuflowtorch/layer/sigmoid.hpp"
#include "nmbuflowtorch/tmp.hpp"

using namespace std;

#include <vector>

int main(int argc, char** argv)
{
  // Sammenligner med utregninger fra https://theneuralblog.com/forward-pass-backpropagation-example/

  nmbuflowtorch::Layer* dense1 = new nmbuflowtorch::layer::Dense(2, 2);

  // Eigen::MatrixXd W = Eigen::MatrixXd(2, 2);
  // W << 0.1, 0.2, 0.3, 0.4;
  // d.set_weights(W);

  // CrossEntropy loss_function = CrossEntropy();

  // Sigmoid s = Sigmoid();

  // Eigen::MatrixXd y = Eigen::MatrixXd(2, 2);
  // y << 0.05, 0.95, 0.05, 0.95;

  // Eigen::MatrixXd X = Eigen::MatrixXd(2, 2);
  // X << 0.1, 0.5, 0.1, 0.5;

  // auto output = d.forward(X);
  // cout << output << endl;

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