#include "nmbuflowtorch/optimizer/sgd.hpp"

namespace nmbuflowtorch::optimizer
{
  Eigen::MatrixXd SGD::get_new_weights(const Eigen::MatrixXd& weights, const Eigen::MatrixXd& grads_wrt_w)
  {
    return weights - learning_rate * grads_wrt_w;

    return weights - 0.001 * grads_wrt_w;
  }
}  // namespace nmbuflowtorch::optimizer