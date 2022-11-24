#include "nmbuflowtorch/loss/cross_entropy.hpp"

namespace nmbuflowtorch::loss
{
  void CrossEntropy::eval(const Matrix& pred, const Matrix& target)
  {
    int n = pred.rows();
    const float epsilon = 1e-8;
    // Matrix ls = -(y.cwiseProduct(p.array().log().matrix()));
    // Matrix rs = (1 - y.array()).matrix().cwiseProduct((1 - p.array()).log().matrix());

    // loss
    loss = -((target.array().cwiseProduct((pred.array() + epsilon).log())).sum() / n);

    // gradient to backpropagate
    gradient_back = -(target.array().cwiseQuotient((pred.array() + epsilon)) / n);

    // https://stats.stackexchange.com/questions/370723/how-to-calculate-the-derivative-of-crossentropy-error-function
    // Matrix back_gradient(Matrix y, Matrix p){
    // Matrix ls = -(y.array() / p.array());
    // Matrix rs = (1 - y.array()) / (1 - p.array());
  };

}  // namespace nmbuflowtorch::loss
