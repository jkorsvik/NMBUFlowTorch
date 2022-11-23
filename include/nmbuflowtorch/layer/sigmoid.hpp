#ifndef NMBUFLOWTORCH_LAYER_SIGMOID_H_
#define NMBUFLOWTORCH_LAYER_SIGMOID_H_

#include "../Layer.hpp"
#include "Eigen/Dense"
namespace nmbuflowtorch::layer
{
  class Sigmoid : public Layer
  {
   public:
    Eigen::MatrixXd forward(const Eigen::MatrixXd& X);
    Eigen::MatrixXd backward(const Eigen::MatrixXd& accumulated_gradients);
  };
}  // namespace nmbuflowtorch::layer
#endif  // NMBUFLOWTORCH_LAYER_SIGMOID_H_
