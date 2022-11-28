#include "nmbuflowtorch/layer/softmax.hpp"

// NOT INCLUDED IN COMPILING YET, NOT TESTED
namespace nmbuflowtorch::layer
{
  void Sigmoid::forward(const Matrix& X)
  {
    layer_input = X;

    layer_output.array() = (X.rowwise() - X.colwise().maxCoeff()).array().exp();
    RowVector z_exp_sum = layer_output.colwise().sum();
    layer_output.array().rowwise() /= z_exp_sum;
  }

  void Sigmoid::backward(const Matrix& X, const Matrix& accumulated_gradients)
  {
    RowVector temp_sum = layer_output.cwiseProduct(accumulated_gradients).colwise().sum();
    gradient_back.array() = layer_output.array().cwiseProduct(accumulated_gradients.array().rowwise() - temp_sum);
  }
}  // namespace nmbuflowtorch::layer