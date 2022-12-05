#ifndef NMBUFLOWTORCH_LAYER_FULLY_CONNECTED_H_
#define NMBUFLOWTORCH_LAYER_FULLY_CONNECTED_H_

#include <string>
#include <vector>

#include "../layer.hpp"
#include "../math_m.hpp"

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

    std::string layer_type = "Dense";

    void init();

   public:
    /// @brief Constructor for Dense layer
    /// @param input_shape: Input shape for the layer
    /// @param units: Number of neurons in the layer
    Dense(const int input_shape, const int units) : input_shape(input_shape), units(units)
    {
      init();
    }

    /// @brief Forward pass for the dense layer
    /// @param X: Input matrix for the layer
    void forward(const Matrix& X);

    /// @brief Backward pass for the dense layer
    /// @param X: Input matrix for the layer
    /// @param accumulated_gradients: Gradients accumulated from previous layers
    void backward(const Matrix& X, const Matrix& accumulated_gradients);

    /// @brief Update the weights and biases of the dense layer
    /// @param opt: Optimizer used to update the parameters
    void update(Optimizer& opt, int epoch) override;

    /// @brief Get the output dimension of the dense layer
    /// @return Output dimension of the dense layer
    int output_dim()
    {
      return units;
    }

    /// @brief Set the weights of the dense layer
    /// @param new_weights: New weights for the layer
    void set_weights(Matrix new_weights)
    {
      // Metode for å overskrive vektene
      weights = new_weights;
    }

    /// @brief Set the bias of the dense layer
    /// @param new_bias: New bias for the layer
    void set_bias(Matrix new_bias)
    {
      bias = new_bias;
    };

    /// @brief Get the type of the layer
    /// @return Type of the layer
    std::string get_layer_type()
    {
      return layer_type;
    };

    /// @brief Get the current parameters of the layer
    /// @return Vector of current parameters of the layer
    std::vector<float> get_parameters() const;

    /// @brief Get the current derivatives of the layer parameters
    /// @return Vector of current derivatives of the layer parameters
    std::vector<float> get_derivatives() const;

    /// @brief Set the parameters of the layer
    /// @param param: Vector of new parameters for the layer
    void set_parameters(const std::vector<float>& param);
  };
}  // namespace nmbuflowtorch::layer
#endif  // NMBUFLOWTORCH_LAYER_FULLY_CONNECTED_H_
