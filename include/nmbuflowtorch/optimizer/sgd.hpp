#ifndef NMBUFLOWTORCH_OPTIMIZER_SGD_H_
#define NMBUFLOWTORCH_OPTIMIZER_SGD_H_

#include "../Optimizer.hpp"
#include "Eigen/Dense"

namespace nmbuflowtorch::optimizer
{
  class SGD : public Optimizer
  {
   private:
    bool weight_init = false;

   public:
    explicit SGD(float learning_rate = 0.01) : Optimizer(learning_rate), weight_init(weight_init)
    {
    }

    Eigen::MatrixXd get_new_weights(const Eigen::MatrixXd& weights, const Eigen::MatrixXd& grads_wrt_w);
  };
}  // namespace nmbuflowtorch::optimizer
#endif  // NMBUFLOWTORCH_OPTIMIZER_SGD_H_
