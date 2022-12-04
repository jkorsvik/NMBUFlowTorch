#include "nmbuflowtorch/layer/conv.hpp"

namespace nmbuflowtorch::layer
{
  void ConvolutionalLayer::init()
  {
    // Set up weights, bias and gradients
    // using helper functions
    weights = random_tensor(input_channels, filter_size, filter_size, filters);
    bias = Vector::Zero(filters);
    grad_weights = Matrix::Zero(input_channels, filter_size * filter_size * filters);
    grad_bias = Vector::Zero(filters);
  }

  void ConvolutionalLayer::forward(const Matrix& X)
  {
    // Perform convolution operation on input data X
    // and store the result in layer_output
    layer_output = convolve(X, weights) + bias;

    // Apply activation function (if any) to layer_output
    layer_output = activation(layer_output);
  }

  void ConvolutionalLayer::backward(const Matrix& X, const Matrix& accumulated_gradients)
  {
    // Compute gradient with respect to weights and bias
    grad_weights = convolve(X.transpose(), accumulated_gradients);
    grad_bias = accumulated_gradients.rowwise().sum();

    // Compute gradient with respect to input X
    gradient_back = convolve(accumulated_gradients, weights.transpose());

    // Apply activation function derivative (if any) to gradient_back
    gradient_back = activation_derivative(gradient_back);
  }
  // Could be made into a math function
  Matrix ConvolutionalLayer::convolve(const Matrix& X, const Matrix& weights)
  {
    // Initialize output matrix with the correct dimensions
    Matrix output = Matrix::Zero(X.rows() - filter_size + 1, X.cols() - filter_size + 1, filters);

    // Perform convolution operation on input data X and weights using Eigen3 operations
    for (int f = 0; f < filters; f++)
    {
      for (int c = 0; c < X.cols() - filter_size + 1; c++)
      {
        for (int r = 0; r < X.rows() - filter_size + 1; r++)
        {
          output.block(r, c, filter_size, filter_size).noalias() +=
              X.block(r, c, filter_size, filter_size) *
              weights.block(0, 0, input_channels, filter_size * filter_size).row(f);
        }
      }
    }

    return output;
  }
