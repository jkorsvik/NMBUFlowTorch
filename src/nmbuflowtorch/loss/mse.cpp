#include "nmbuflowtorch/loss/mse.hpp"
namespace nmbuflowtorch::loss
{
  void MSE::eval(const Matrix& pred, const Matrix& target)
  {
    int n = pred.cols();
    // forward: L = sum{ (p-y).*(p-y) } / n
    Matrix diff = pred - target;
    loss = diff.cwiseProduct(diff).sum();
    loss /= n;
    // backward: d(L)/d(p) = (p-y)*2/n
    gradient_back = diff * 2 / n;
  }
}  // namespace nmbuflowtorch::loss