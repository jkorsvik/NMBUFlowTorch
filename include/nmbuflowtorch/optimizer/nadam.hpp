#ifndef NMBUFLOWTORCH_OPTIMIZER_NADAM_H_
#define NMBUFLOWTORCH_OPTIMIZER_NADAM_H_

#include "../optimizer.hpp"

namespace nmbuflowtorch::optimizer
{
  /// @brief Nadam optimizer
  class Nadam : public Optimizer
  {
   private:
    std::unordered_map<const float*, Vector> m_map;  // mean map to pointers
    std::unordered_map<const float*, Vector> v_map;  // variance map to pointers
    float beta1;
    float beta2;
    float weight_decay;
    float momentum_decay;
    float epsilon;  // epsilon for numerical stability
    int t = 0;      // time step

   public:
    explicit Nadam(
        float learning_rate = 2e-3,
        float beta1 = 0.9,
        float beta2 = 0.999,
        float weight_decay = 0.0,
        float momentum_decay = 4e-3,
        float epsilon = 1e-8)
        : Optimizer(learning_rate),
          beta1(beta1),
          beta2(beta2),
          weight_decay(weight_decay),
          momentum_decay(momentum_decay),
          epsilon(epsilon)
    {
    }

    virtual void update(Vector::AlignedMapType& w, Vector::ConstAlignedMapType& dw);
  };
}  // namespace nmbuflowtorch::optimizer
#endif  // NMBUFLOWTORCH_OPTIMIZER_NADAM_H_