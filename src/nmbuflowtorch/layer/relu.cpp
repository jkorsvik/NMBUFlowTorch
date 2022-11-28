#include "nmbuflowtorch/layer/relu.hpp"

namespace nmbuflowtorch::layer
{

  void ReLU::forward(const Matrix& X)
  {
      layer_input = X;
    layer_output = X.cwiseMax(0.0);
  }

  void ReLU::backward(const Matrix& X, const Matrix& accumulated_gradients)
  {
    Matrix positive = (X.array() > 0.0).cast<float>();
    gradient_back = accumulated_gradients.cwiseProduct(positive);
  }
}  // namespace nmbuflowtorch::layer