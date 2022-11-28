#ifndef NMBUFLOWTORCH_LOSS_MSE_H_
#define NMBUFLOWTORCH_LOSS_MSE_H_

#include "../loss.hpp"

namespace nmbuflowtorch::loss
{
  class MSE : public Loss
  {
   public:
    void eval(const Matrix& pred, const Matrix& target);
  };
}  // namespace nmbuflowtorch::loss
#endif  // NMBUFLOWTORCH_LOSS_MSE_H_
