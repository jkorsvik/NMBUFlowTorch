#include "nmbuflowtorch/loss/cross_entropy_loss.hpp"

namespace nmbuflowtorch::loss
{
  Eigen::MatrixXd loss(Eigen::MatrixXd y, Eigen::MatrixXd p)
  {
    Eigen::MatrixXd ls = -(y.cwiseProduct(p.array().log().matrix()));

    Eigen::MatrixXd rs = (1 - y.array()).matrix().cwiseProduct((1 - p.array()).log().matrix());

    return ls - rs;
  };

  // https://stats.stackexchange.com/questions/370723/how-to-calculate-the-derivative-of-crossentropy-error-function
  Eigen::MatrixXd back_gradient(Eigen::MatrixXd y, Eigen::MatrixXd p)
  {
    Eigen::MatrixXd ls = -(y.array() / p.array());
    Eigen::MatrixXd rs = (1 - y.array()) / (1 - p.array());

    return ls + rs;
  };

}  // namespace nmbuflowtorch::loss
