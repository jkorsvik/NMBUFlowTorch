#ifndef NMBUFLOWTORCH_LAYER_SIGMOID_H_
#define NMBUFLOWTORCH_LAYER_SIGMOID_H_

#include "../layer.hpp"

namespace nmbuflowtorch::layer
{
  class Sigmoid : public Layer
  {
   private:
    std::string layer_type = "Sigmoid";

   public:
    /// @brief Apply sigmoid function to input
    /// @param X : Input data
    void forward(const Matrix& X);

    /// @brief Backward pass of sigmoid layer
    /// @param X : Input data
    /// @param accumulated_gradients : Gradients accumulated from previous layers
    void backward(const Matrix& X, const Matrix& accumulated_gradients);
  };
}  // namespace nmbuflowtorch::layer
#endif  // NMBUFLOWTORCH_LAYER_SIGMOID_H_
