#ifndef NMBUFLOWTORCH_LAYER_SOFTMAX_H_
#define NMBUFLOWTORCH_LAYER_SOFTMAX_H_

#include "../layer.h"
namespace nmbuflowtorch::layer
{
  class Softmax : public Layer
  {
   public:
    void forward(const Matrix& bottom);
    void backward(const Matrix& bottom, const Matrix& grad_top);
  };
}  // namespace nmbuflowtorch::layer
#endif  // NMBUFLOWTORCH_LAYER_SOFTMAX_H_
