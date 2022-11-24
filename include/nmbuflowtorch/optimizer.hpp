#ifndef NMBUFLOWTORCH_OPTIMIZER_H_
#define NMBUFLOWTORCH_OPTIMIZER_H_

#include <unordered_map>

#include "./definitions.hpp"

namespace nmbuflowtorch
{
  class Optimizer
  {
   protected:             // Private but accesible to derived classes
    float learning_rate;  // learning rate

   public:
    explicit Optimizer(float learning_rate = 0.01) : learning_rate(learning_rate)
    {
    }
    virtual ~Optimizer()
    {
    }
    virtual void update(Vector::AlignedMapType& w, Vector::ConstAlignedMapType& dw) = 0;
  };
};      // namespace nmbuflowtorch
#endif  // NMBUFLOWTORCH_OPTIMIZER_H_
