#ifndef NMBUFLOWTORCH_LOSS_H_
#define NMBUFLOWTORCH_LOSS_H_

//#include "./utils.h"

#include "Eigen/Dense"
namespace nmbuflowtorch
{
  class Loss
  {
   public:
    virtual ~Loss() = default;

    Eigen::MatrixXd loss(const Eigen::MatrixXd& pred, const Eigen::MatrixXd& target);

    const Eigen::MatrixXd& back_gradient(const Eigen::MatrixXd& pred, const Eigen::MatrixXd& target);
  };
}  // namespace nmbuflowtorch
#endif  // NMBUFLOWTORCH_LOSS_H_
