#ifndef NMBUFLOWTORCH_OPTIMIZER_H_
#define NMBUFLOWTORCH_OPTIMIZER_H_

#include "./utils.h"
namespace nmbuflowtorch {
class Optimizer {
 protected:
  float lr;  // learning rate
  float decay;  // weight decay factor (default: 0)

 public:
  explicit Optimizer(float lr = 0.01, float decay = 0.0) :
                     lr(lr), decay(decay) {}
  virtual ~Optimizer() {}

  virtual void update(Vector::AlignedMapType& w,
                      Vector::ConstAlignedMapType& dw) = 0;
};
}  // namespace nmbuflowtorch
#endif  // NMBUFLOWTORCH_OPTIMIZER_H_
