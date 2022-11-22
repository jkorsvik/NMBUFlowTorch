#ifndef NMBUFLOWTORCH_LAYER_SIGMOID_H_
#define NMBUFLOWTORCH_LAYER_SIGMOID_H_

#include "../layer.h"

class Sigmoid : public Layer {
 public:
  void forward(const Matrix& bottom);
  void backward(const Matrix& bottom, const Matrix& grad_top);
};

#endif  // NMBUFLOWTORCH_LAYER_SIGMOID_H_
