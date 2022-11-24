#ifndef NMBUFLOWTORCH_NETWORK_H_
#define NMBUFLOWTORCH_NETWORK_H_

#include <stdlib.h>

#include <vector>

#include "./definitions.hpp"
#include "./layer.hpp"
#include "./loss.hpp"
#include "./optimizer.hpp"

namespace nmbuflowtorch
{

  class Network
  {
   private:
    std::vector<Layer*> layers;  // layer pointers
    Loss* loss;                  // loss pointer

    // make public?

   public:
    Network() : loss(NULL){};
    ~Network()
    {
      for (int i = 0; i < layers.size(); i++)
      {
        // delete value and pointers
        delete layers[i];
      }
      if (loss)
      {
        // delete value and pointer
        delete loss;
      }
    }
    void add_layer(Layer* layer)
    {
      layers.push_back(layer);
    }
    void add_loss(Loss* loss_in)
    {
      loss = loss_in;
    }

    // Wrapper functions for forward, backward and update
    // TODO: add fit or train function to handle epochs and batches

    void forward(const Matrix& X);

    /// @brief Using X input and y target, back propagates the loss gradient
    /// @param X
    /// @param y
    void backward(const Matrix& X, const Matrix& y);

    /// @brief
    /// @param opt : Optimizer object reference
    void update(Optimizer& opt);

    void train_batch(){};
    void fit(){};
    void predict(){};

    /// @brief Returns the value from the last layer
    /// @return Matrix&
    const Matrix& output()
    {
      return layers.back()->output();
    }
    float get_loss()
    {
      return loss->output();
    }
    /// Get the serialized layer parameters
    std::vector<std::vector<float>> get_parameters() const;
    /// Set the layer parameters
    void set_parameters(const std::vector<std::vector<float>>& param);
    /// Get the serialized derivatives of layer parameters
    std::vector<std::vector<float>> get_derivatives() const;
    /// Debugging tool to check parameter gradients
    void check_gradient(const Matrix& input, const Matrix& target, int n_points, int seed = -1);
  };
}  // namespace nmbuflowtorch
#endif  // NMBUFLOWTORCH_NETWORK_H_
