#include "nmbuflowtorch/loss/mse.hpp"

namespace nmbuflowtorch::loss
{
  void MSE::eval(const Matrix& pred, const Matrix& target)
  {
    int n = pred.cols();

    Matrix diff = pred - target;
    loss = diff.cwiseProduct(diff).sum();
    loss *= (0.5 / n);

    gradient_back = diff / n;
  }
}  // namespace nmbuflowtorch::loss