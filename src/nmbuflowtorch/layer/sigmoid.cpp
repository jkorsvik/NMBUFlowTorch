#include "nmbuflowtorch/layer/sigmoid.hpp"

namespace nmbuflowtorch::layer
{

  void Sigmoid::forward(const Matrix& X)
  {
    layer_input = X;
    layer_output = 1.0 / (1.0 + (-layer_input).array().exp());

    //
    // X.array() = 1.0 / (1.0 + (-X).array().exp());
    // fast sigmoid approximation f(x) = x / (1 + abs(x))
  }

  void Sigmoid::backward(const Matrix& X, const Matrix& accumulated_gradients)
  {
    Matrix da_dz = layer_output.array().cwiseProduct(1.0 - layer_output.array());
    da_dz.resize(accumulated_gradients.rows(), accumulated_gradients.cols());  // Mulig unødvendig
    gradient_back = accumulated_gradients.cwiseProduct(da_dz);
  }

}  // namespace nmbuflowtorch::layer
