#ifndef NMBUFLOWTORCH_LOSS_CROSS_ENTROPY_LOSS_H_
#define NMBUFLOWTORCH_LOSS_CROSS_ENTROPY_LOSS_H_

#include "../loss.hpp"
#include "Eigen/Dense"

namespace nmbuflowtorch::loss
{
  class CrossEntropy : public Loss
  {
   public:
    Eigen::MatrixXd loss(const Eigen::MatrixXd& pred, const Eigen::MatrixXd& target);
    Eigen::MatrixXd back_gradient(const Eigen::MatrixXd& pred, const Eigen::MatrixXd& target);
  };
}  // namespace nmbuflowtorch::loss

#endif  // NMBUFLOWTORCH_LOSS_CROSS_ENTROPY_LOSS_H_
