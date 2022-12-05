#ifndef NMBUFLOWTORCH_LAYER_H_
#define NMBUFLOWTORCH_LAYER_H_

#include <vector>

#include "./definitions.hpp"
#include "./optimizer.hpp"

namespace nmbuflowtorch
{
  /// @brief The base layer class for neural network models.
  class Layer
  {
   protected:
    Matrix layer_input;
    Matrix layer_output;
    Matrix gradient_back;

    std::string layer_type = "Virtual Layer";

   public:
    virtual ~Layer()
    {
    }
    /// @brief Forward pass of the layer.
    /// @param X The input to the layer.
    virtual void forward(const Matrix& X) = 0;

    /// @brief Backward pass of the layer.
    /// @param X The input to the layer.
    /// @param accumulated_gradients The gradient accumulated so far in the network.
    virtual void backward(const Matrix& X, const Matrix& accumulated_gradients) = 0;

    /// @brief Updates the layer's parameters using the given optimizer.
    /// @param opt The optimizer to use for updating the parameters.
    /// @param learning_rate The learning rate to use for the update.
    virtual void update(Optimizer& opt, int epoch = 0)
    {
    }

    virtual const Matrix& back_gradient()
    {
      return gradient_back;
    }
    virtual const Matrix& last_input()
    {
      return layer_input;
    }

    virtual const Matrix& output()
    {
      return layer_output;
    }

    /// @brief Returns the output dimension of the layer.
    virtual int output_dim()
    {
      return -1;
    }

    /// @brief Returns the parameters of the layer.
    virtual std::vector<float> get_parameters() const
    {
      return std::vector<float>();
    }

    /// @brief Sets the parameters of the layer.
    /// @param param The new parameters for the layer.
    virtual void set_parameters(const std::vector<float>& param)
    {
    }

    /// @brief Returns the derivatives of the parameters of the layer.
    virtual std::vector<float> get_derivatives() const
    {
      return std::vector<float>();
    }

    /// @brief Returns the type of the layer.
    virtual std::string get_layer_type() const
    {
      return layer_type;
    };
  };

}  // namespace nmbuflowtorch
#endif  // LAYER_H_
