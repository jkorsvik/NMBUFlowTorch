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

    /// @brief Evaluates the loss function given the predicted and target values.
    /// @param pred The predicted values.
    /// @param target The target values.
    virtual void eval(const Matrix& pred, const Matrix& target) = 0;

    /// @brief Returns the gradient with respect to the input of the loss function.
    virtual const Matrix& back_gradient()
    {
      return gradient_back;
    }

    /// @brief Returns the value of the loss.
    virtual float output()
    {
      return loss;
    }
  };
}  // namespace nmbuflowtorch
#endif  // NMBUFLOWTORCH_LOSS_H_
