#ifndef NMBUFLOWTORCH_LAYER_SOFTMAX_H_
#define NMBUFLOWTORCH_LAYER_SOFTMAX_H_

#include "../layer.hpp"
// NOT INCLUDED IN COMPILING YET, NOT TESTED
namespace nmbuflowtorch::layer
{
  class Softmax : public Layer
  {
   private:
    std::string layer_type = "Softmax";

   public:
    void forward(const Matrix& X);
    void backward(const Matrix& X, const Matrix& accumulated_gradients);
  };
}  // namespace nmbuflowtorch::layer
#endif  // NMBUFLOWTORCH_LAYER_SOFTMAX_H_
