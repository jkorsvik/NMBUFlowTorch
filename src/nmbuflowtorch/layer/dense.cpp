#include "nmbuflowtorch/layer/dense.hpp"

#include "nmbuflowtorch/Optimizer.hpp"

#

namespace nmbuflowtorch::layer
{

  void Dense::init()
  {
    weights = Eigen::MatrixXd::Random(input_shape, units);  // TODO: annen initialisering? Tror det er -1 til 1 her
    bias = Eigen::MatrixXd::Zero(1, units);
  }

  Eigen::MatrixXd Dense::forward(const Eigen::MatrixXd& X)
  {
    layer_input = X;                       // Holder på input til backward passet
    layer_output = layer_input * weights;  // TODO: Plusse på bias WX + B
    layer_output += bias;                  // Eigen::Matrix3Xf::Ones(1,)
    // mtx += Eigen::Matrix3Xf::Ones(3,4);
    return layer_output;
  }

  Eigen::MatrixXd Dense::backward(Eigen::MatrixXd& accumulated_gradients)
  {
    // Eigen::MatrixXd prev_weights = weights;

    Eigen::MatrixXd grad_weights = layer_input.transpose() * accumulated_gradients;
    Eigen::MatrixXd grad_bias = accumulated_gradients.colwise().sum();

    accumulated_gradients = accumulated_gradients * weights.transpose();

    // Oppdaterer vektene
    weights = weight_optimizer.get_new_weights(weights, grad_weights);
    bias = bias_optimizer.get_new_weights(bias, grad_bias);

    // accumulated_gradients = accumulated_gradients * prev_weights.transpose();

    return accumulated_gradients;
  }

}  // namespace nmbuflowtorch::layer