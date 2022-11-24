#ifndef NMBUFLOWTORCH_LAYER_SIGMOID_H_
#define NMBUFLOWTORCH_LAYER_SIGMOID_H_

#include "../layer.hpp"

namespace nmbuflowtorch::layer
{
  class Sigmoid : public Layer
  {
   public:
    void forward(const Matrix& X);
    void backward(const Matrix& X, const Matrix& accumulated_gradients);
  };
}  // namespace nmbuflowtorch::layer
#endif  // NMBUFLOWTORCH_LAYER_SIGMOID_H_
