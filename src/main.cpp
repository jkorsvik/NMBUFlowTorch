#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "nmbuflowtorch/layer.h"
#include "nmbuflowtorch/layer/ave_pooling.h"
#include "nmbuflowtorch/layer/conv.h"
#include "nmbuflowtorch/layer/fully_connected.h"
#include "nmbuflowtorch/layer/max_pooling.h"
#include "nmbuflowtorch/layer/relu.h"
#include "nmbuflowtorch/layer/sigmoid.h"
#include "nmbuflowtorch/layer/softmax.h"
#include "nmbuflowtorch/loss.h"
#include "nmbuflowtorch/loss/cross_entropy_loss.h"
#include "nmbuflowtorch/loss/mse_loss.h"
#include "nmbuflowtorch/mnist.h"
#include "nmbuflowtorch/network.h"
#include "nmbuflowtorch/optimizer.h"
#include "nmbuflowtorch/optimizer/sgd.h"
#include "nmbuflowtorch/tmp.hpp"

// using namespace nmbuflowtorch::layer;
// using namespace nmbuflowtoch::loss;
// using namespace nmbuflowtorch::optimizer;
// using namespace nmbuflowtorch;

int main(int argc, char** argv)
{
  nmbuflowtorch::MNIST dataset("../data/mnist/");
  dataset.read();
  int n_train = dataset.train_data.cols();
  int dim_in = dataset.train_data.rows();
  std::cout << "mnist train number: " << n_train << std::endl;
  std::cout << "mnist test number: " << dataset.test_labels.cols() << std::endl;
  // dnn
  nmbuflowtorch::Network dnn;
  nmbuflowtorch::Layer* conv1 = new nmbuflowtorch::layer::Conv(1, 28, 28, 4, 5, 5, 2, 2, 2);
  nmbuflowtorch::Layer* pool1 = new nmbuflowtorch::layer::MaxPooling(4, 14, 14, 2, 2, 2);
  nmbuflowtorch::Layer* conv2 = new nmbuflowtorch::layer::Conv(4, 7, 7, 16, 5, 5, 1, 2, 2);
  nmbuflowtorch::Layer* pool2 = new nmbuflowtorch::layer::MaxPooling(16, 7, 7, 2, 2, 2);
  nmbuflowtorch::Layer* fc3 = new nmbuflowtorch::layer::FullyConnected(pool2->output_dim(), 32);
  nmbuflowtorch::Layer* fc4 = new nmbuflowtorch::layer::FullyConnected(32, 10);
  nmbuflowtorch::Layer* relu1 = new nmbuflowtorch::layer::ReLU;
  nmbuflowtorch::Layer* relu2 = new nmbuflowtorch::layer::ReLU;
  nmbuflowtorch::Layer* relu3 = new nmbuflowtorch::layer::ReLU;
  nmbuflowtorch::Layer* softmax = new nmbuflowtorch::layer::Softmax;
  dnn.add_layer(conv1);
  dnn.add_layer(relu1);
  dnn.add_layer(pool1);
  dnn.add_layer(conv2);
  dnn.add_layer(relu2);
  dnn.add_layer(pool2);
  dnn.add_layer(fc3);
  dnn.add_layer(relu3);
  dnn.add_layer(fc4);
  dnn.add_layer(softmax);
  // loss
  nmbuflowtorch::Loss* loss = new nmbuflowtorch::loss::CrossEntropy;
  dnn.add_loss(loss);
  // train & test
  nmbuflowtorch::optimizer::SGD opt(0.001, 5e-4, 0.9, true);
  // SGD opt(0.001);
  const int n_epoch = 5;
  const int batch_size = 128;
  for (int epoch = 0; epoch < n_epoch; epoch++)
  {
    nmbuflowtorch::shuffle_data(dataset.train_data, dataset.train_labels);
    for (int start_idx = 0; start_idx < n_train; start_idx += batch_size)
    {
      int ith_batch = start_idx / batch_size;
      Matrix x_batch = dataset.train_data.block(0, start_idx, dim_in, std::min(batch_size, n_train - start_idx));
      Matrix label_batch = dataset.train_labels.block(0, start_idx, 1, std::min(batch_size, n_train - start_idx));
      Matrix target_batch = nmbuflowtorch::one_hot_encode(label_batch, 10);
      if (false && ith_batch % 10 == 1)
      {
        std::cout << ith_batch << "-th grad: " << std::endl;
        dnn.check_gradient(x_batch, target_batch, 10);
      }
      dnn.forward(x_batch);
      dnn.backward(x_batch, target_batch);
      // display
      if (ith_batch % 50 == 0)
      {
        std::cout << ith_batch << "-th batch, loss: " << dnn.get_loss() << std::endl;
      }
      // optimize
      dnn.update(opt);
    }
    // test
    dnn.forward(dataset.test_data);
    float acc = nmbuflowtorch::compute_accuracy(dnn.output(), dataset.test_labels);
    std::cout << std::endl;
    std::cout << epoch + 1 << "-th epoch, test acc: " << acc << std::endl;
    std::cout << std::endl;
  }
  return 0;
}