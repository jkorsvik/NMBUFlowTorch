#ifndef NMBUFLOWTORCH_LAYER_FULLY_CONNECTED_H_
#define NMBUFLOWTORCH_LAYER_FULLY_CONNECTED_H_

#include <vector>

#include "../layer.hpp"

namespace nmbuflowtorch::layer
{
  class Dense : public Layer
  {
   private:
    const int input_shape;
    const int units;  // Number of neurons in the layer ae output shape

    Matrix weights;
    Vector bias;
    Matrix grad_weights;  // gradient --> weights
    Vector grad_bias;     // gradient --> bias

    void init();

   public:
    Dense(const int input_shape, const int units) : input_shape(input_shape), units(units)
    {
      init();
    }

    void forward(const Matrix& X);
    // TODO : Add backward function with const reference argument
    void backward(const Matrix& X, const Matrix& accumulated_gradients);

    void update(Optimizer& opt) override;

    int output_dim() final
    {
      return units;
    }

    void set_weights(Matrix new_weights)
    {
      // Metode for å overskrive vektene
      weights = new_weights;
    }

    void set_bias(Matrix new_bias)
    {
      bias = new_bias;
    };

    std::vector<float> get_parameters() const;
    std::vector<float> get_derivatives() const;
    void set_parameters(const std::vector<float>& param);
  };
}  // namespace nmbuflowtorch::layer
#endif  // NMBUFLOWTORCH_LAYER_FULLY_CONNECTED_H_
