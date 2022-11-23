#include "nmbuflowtorch/layer/sigmoid.hpp"

namespace nmbuflowtorch::layer
{

  Eigen::MatrixXd Sigmoid::forward(const Eigen::MatrixXd& X)
  {
    // a = 1 / (1 + exp(-z))
    layer_input = X;
    layer_output = 1.0 / (1.0 + (-layer_input).array().exp());
    return layer_output;
    //
    // X.array() = 1.0 / (1.0 + (-X).array().exp());
    // fast sigmoid approximation f(x) = x / (1 + abs(x))
  }

  Eigen::MatrixXd Sigmoid::backward(const Eigen::MatrixXd& accumulated_gradients)
  {
    // d(L)/d(z_i) = d(L)/d(a_i) * d(a_i)/d(z_i)
    // d(a_i)/d(z_i) = a_i * (1-a_i)

    Eigen::MatrixXd da_dz = layer_output.array().cwiseProduct(1.0 - layer_output.array());
    return accumulated_gradients.cwiseProduct(da_dz);
  }

}  // namespace nmbuflowtorch::layer
