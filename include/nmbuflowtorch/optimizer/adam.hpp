#ifndef NMBUFLOWTORCH_OPTIMIZER_ADAM_H_
#define NMBUFLOWTORCH_OPTIMIZER_ADAM_H_

#include "../optimizer.hpp"

namespace nmbuflowtorch::optimizer
{
  class Adam : public Optimizer
  {
   private:
    std::unordered_map<const float*, Vector> m_map;  // mean map to pointers
    std::unordered_map<const float*, Vector> v_map;  // variance map to pointers
    float beta1;
    float beta2;

    float epsilon = 1e-8;  // epsilon for numerical stability
    int t = 0;             // time step

   public:
    explicit Adam(float learning_rate = 0.001, float beta1 = 0.9, float beta2 = 0.999)
        : Optimizer(learning_rate),
          beta1(beta1),
          beta2(beta2)
    {
    }

    virtual void update(Vector::AlignedMapType& w, Vector::ConstAlignedMapType& dw);
  };
}  // namespace nmbuflowtorch::optimizer
#endif  // NMBUFLOWTORCH_OPTIMIZER_ADAM_H_