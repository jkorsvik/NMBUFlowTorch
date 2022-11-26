#ifndef NMBUFLOWTORCH_LAYER_SIGMOID_H_
#define NMBUFLOWTORCH_LAYER_SIGMOID_H_

#include "../layer.hpp"

#include <string>


namespace nmbuflowtorch::layer
{
  class Sigmoid : public Layer
  {

    private:
      std::string layer_type = "Sigmoid";

   public:

    void forward(const Matrix& X);
    void backward(const Matrix& X, const Matrix& accumulated_gradients);

    std::string get_layer_type() {
      return layer_type;
    };
  };
}  // namespace nmbuflowtorch::layer
#endif  // NMBUFLOWTORCH_LAYER_SIGMOID_H_
