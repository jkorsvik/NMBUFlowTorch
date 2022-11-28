#include "nmbuflowtorch/layer/relu.hpp"

namespace nmbuflowtorch::layer
{

  void ReLU::forward(const Matrix& X)
  {
    // a = z*(z>0)
    layer_input = X;
    layer_output = X.cwiseMax(0.0);
  }

  void ReLU::backward(const Matrix& X, const Matrix& accumulated_gradients)
  {
    // d(L)/d(z_i) = d(L)/d(a_i) * d(a_i)/d(z_i)
    //             = d(L)/d(a_i) * 1*(z_i>0)
    Matrix positive = (X.array() > 0.0).cast<float>();
    gradient_back = accumulated_gradients.cwiseProduct(positive);
  }
}  // namespace nmbuflowtorch::layer