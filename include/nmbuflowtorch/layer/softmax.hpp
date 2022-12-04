#ifndef NMBUFLOWTORCH_LAYER_SOFTMAX_H_
#define NMBUFLOWTORCH_LAYER_SOFTMAX_H_

#include "../layer.hpp"
// NOT INCLUDED IN COMPILING YET, NOT TESTED
namespace nmbuflowtorch::layer
{
  /// @brief Implements the forward and backward propagation of the softmax function
  class Softmax : public Layer
  {
   private:
    std::string layer_type = "Softmax";

   public:
    /// @brief Forward propagation of the softmax function
    /// @param X : Input data
    void forward(const Matrix& X);

    /// @brief Backward propagation of the softmax function
    /// @param X : Input data
    /// @param accumulated_gradients : Gradients accumulated from previous layers
    void backward(const Matrix& X, const Matrix& accumulated_gradients);
  };
}  // namespace nmbuflowtorch::layer
#endif  // NMBUFLOWTORCH_LAYER_SOFTMAX_H_
