#ifndef NMBUFLOWTORCH_OPTIMIZER_H_
#define NMBUFLOWTORCH_OPTIMIZER_H_

#include "Eigen/Dense"
namespace nmbuflowtorch
{
  class Optimizer
  {
   protected:             // Private but accesible to derived classes
    float learning_rate;  // learning rate

   public:
    explicit Optimizer(float learning_rate = 0.01) : learning_rate(learning_rate)
    {
    }
    virtual ~Optimizer()
    {
    }

    Eigen::MatrixXd get_new_weights(const Eigen::MatrixXd& weights, const Eigen::MatrixXd& grads_wrt_w);
  };
}  // namespace nmbuflowtorch
#endif  // NMBUFLOWTORCH_OPTIMIZER_H_
