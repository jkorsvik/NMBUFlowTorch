#ifndef NMBUFLOWTORCH_LOSS_CROSS_ENTROPY_LOSS_H_
#define NMBUFLOWTORCH_LOSS_CROSS_ENTROPY_LOSS_H_

#include "../loss.h"

namespace nmbuflowtorch::loss
{
  class CrossEntropy : public Loss
  {
   public:
    void evaluate(const Matrix& pred, const Matrix& target);
  };
}  // namespace nmbuflowtorch::loss

#endif  // NMBUFLOWTORCH_LOSS_CROSS_ENTROPY_LOSS_H_
