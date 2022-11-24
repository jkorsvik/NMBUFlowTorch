#ifndef NMBUFLOWTORCH_LOSS_CROSS_ENTROPY_LOSS_H_
#define NMBUFLOWTORCH_LOSS_CROSS_ENTROPY_LOSS_H_

#include "../loss.hpp"

namespace nmbuflowtorch::loss
{
  class CrossEntropy : public Loss
  {
   public:
    void eval(const Matrix& pred, const Matrix& target);
    //  Matrix back_gradient(const Matrix& pred, const Matrix& target);
  };
}  // namespace nmbuflowtorch::loss

#endif  // NMBUFLOWTORCH_LOSS_CROSS_ENTROPY_LOSS_H_
