#ifndef NMBUFLOWTORCH_LAYER_RELU_H_
#define NMBUFLOWTORCH_LAYER_RELU_H_

#include "../layer.hpp"

namespace nmbuflowtorch::layer
{
  class ReLU : public Layer
  {
   private:
    std::string layer_type = "ReLu";

   public:
    void forward(const Matrix& X);
    void backward(const Matrix& X, const Matrix& accumulated_gradients);
  };
}  // namespace nmbuflowtorch::layer
#endif  // NMBUFLOWTORCH_LAYER_RELU_H_
