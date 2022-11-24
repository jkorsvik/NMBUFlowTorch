#ifndef NMBUFLOWTORCH_OPTIMIZER_SGD_H_
#define NMBUFLOWTORCH_OPTIMIZER_SGD_H_

#include "../optimizer.hpp"

namespace nmbuflowtorch::optimizer
{
  class SGD : public Optimizer
  {
   private:
    std::unordered_map<const float*, Vector> v_map;  // velocity map to pointers
   public:
    explicit SGD(float learning_rate = 0.01) : Optimizer(learning_rate)
    {
    }

    virtual void update(Vector::AlignedMapType& w, Vector::ConstAlignedMapType& dw);
  };
}  // namespace nmbuflowtorch::optimizer
#endif  // NMBUFLOWTORCH_OPTIMIZER_SGD_H_
