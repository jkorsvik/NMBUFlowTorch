#ifndef NMBUFLOWTORCH_LOSS_H_
#define NMBUFLOWTORCH_LOSS_H_

#include "./definitions.hpp"

namespace nmbuflowtorch
{
  class Loss
  {
   protected:
    float loss;
    // const float epsilon = 1e-8;
    Matrix gradient_back;

   public:
    virtual ~Loss()
    {
    }

    virtual void eval(const Matrix& pred, const Matrix& target) = 0;

    virtual const Matrix& back_gradient()
    {
      return gradient_back;
    }

    virtual float output()
    {
      return loss;
    }
  };
}  // namespace nmbuflowtorch
#endif  // NMBUFLOWTORCH_LOSS_H_
