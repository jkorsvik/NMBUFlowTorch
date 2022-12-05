#ifndef NMBUFLOWTORCH_OPTIMIZER_H_
#define NMBUFLOWTORCH_OPTIMIZER_H_

#include <unordered_map>

#include "./definitions.hpp"

/**

@class Optimizer

@brief Abstract base class for optimizers in NMBUFLOWTORCH

This class provides the basic structure and common parameters used by all optimizers.

@param learning_rate : Learning rate for the optimizer

@fn update : Virtual function that should be implemented by derived classes. This function updates the model parameters given
the gradient of the parameters.
*/
namespace nmbuflowtorch
{
  class Optimizer
  {
   protected:              // Private but accesible to derived classes
    float learning_rate;   // learning rate
    float epsilon = 1e-8;  // epsilon for numerical stability
    int t = 0;             // time step

   public:
    /**

  @brief Constructor for Optimizer
  @param learning_rate : Learning rate for the optimizer. Default value is 0.01
  */
    explicit Optimizer(float learning_rate = 0.01) : learning_rate(learning_rate)
    {
    }
    /**
    @brief Destructor for Optimizer
    */
    virtual ~Optimizer()
    {
    }

    ///@brief Virtual function that should be implemented by derived classes.
    /// This function updates the model parameters given the gradient of the parameters.
    ///@param w : Aligned map of the model parameters to be updated
    ///@param dw : Const aligned map of the gradient of the model parameters
    virtual void update(Vector::AlignedMapType& w, Vector::ConstAlignedMapType& dw, int epoch = 0) = 0;
  };
};      // namespace nmbuflowtorch
#endif  // NMBUFLOWTORCH_OPTIMIZER_H_
