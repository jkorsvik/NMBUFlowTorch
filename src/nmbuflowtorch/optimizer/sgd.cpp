#include "nmbuflowtorch/optimizer/sgd.hpp"
namespace nmbuflowtorch::optimizer
{
  Eigen::MatrixXd SGD::get_new_weights(Eigen::MatrixXd weights, Eigen::MatrixXd grads_wrt_w)
  {
    weight_init = weight_init;
    if (weight_init == false)
    {
      weight_update = Eigen::MatrixXd::Zero(weights.rows(), weights.cols());
      weight_init = true;
    }

    return weights - 0.001 * grads_wrt_w;
  }
}  // namespace nmbuflowtorch::optimizer