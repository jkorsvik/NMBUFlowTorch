#ifndef NMBUFLOWTORCH_LAYER_CONV_H_
#define NMBUFLOWTORCH_LAYER_CONV_H_

#include "../layer.hpp"

namespace nmbuflowtorch::layer
{
  class Convolutional : public Layer
  {
   private:
    std::string layer_type = "Convolutional";

    // Private members for storing filter weights, bias, and gradients
    Matrix weights;
    Vector bias;
    Matrix grad_weights;
    Vector grad_bias;

    // Private member for storing convolutional hyperparameters
    int stride;
    int padding;

    // Private helper function for initializing weights and bias
    void init();

   public:
    // Constructor for convolutional layer
    // Takes in the number of filters, the size of the filters, stride, and padding
    Convolutional(const int num_filters, const int filter_size, const int stride, const int padding)
        : num_filters(num_filters),
          filter_size(filter_size),
          stride(stride),
          padding(padding)
    {
      init();
    }

    // Forward propagation for convolutional layer
    // Takes in input data and computes the convolutional operation
    void forward(const Matrix& X);

    // Backward propagation for convolutional layer
    // Takes in input data and accumulated gradients, and computes the gradient with respect to the input data
    void backward(const Matrix& X, const Matrix& accumulated_gradients);

    // Overridden update function for convolutional layer
    // Takes in an Optimizer object and updates the weights and bias using the gradients
    void update(Optimizer& opt) override;

    // Output dimension function for convolutional layer
    // Returns the number of filters as the output dimension
    int output_dim()
    {
      return num_filters;
    }

    // Getter and setter functions for weights and bias
    void set_weights(Matrix new_weights)
    {
      weights = new_weights;
    }
    void set_bias(Vector new_bias)
    {
      bias = new_bias;
    }
    Matrix get_weights()
    {
      return weights;
    }
    Vector get_bias()
    {
      return bias;
    }

    // Getter function for layer type
    std::string get_layer_type()
    {
      return layer_type;
    }

    // Functions for getting and setting layer parameters and derivatives
    std::vector<float> get_parameters() const;
    std::vector<float> get_derivatives() const;
    void set_parameters(const std::vector<float>& param);
  };
}  // namespace nmbuflowtorch::layer

#endif  // NMBUFLOWTORCH_LAYER_CONV_H_
