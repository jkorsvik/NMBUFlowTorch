/**

@file
@brief Network class representing a sequence of layers.
This class implements a network consisting of a sequence of layers.
It provides functions for forward and backward propagation, training, and
predictions.
*/

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
  ///@brief The Network class is a class for creating and managing a neural network. It has functions for adding layers,
  /// losses, and optimizers to the network, as well as functions for training and predicting using the network. The summary
  /// function prints a summary of the network, including the type and output shape of each layer. The fit function trains
  /// the network on a given dataset, with options for specifying the number of epochs, batch size, and verbosity level. The
  /// predict function uses the network to make predictions on a given dataset. The get_parameters and get_derivatives
  /// functions return the serialized layer parameters and derivatives of layer parameters, respectively. The
  /// get_layer_output and get_layer_weight functions return the output and weight matrices of a specified layer.
  {
   private:
    std::vector<Layer*> layers;  // layer pointers
    Loss* loss;                  // loss pointer
    Optimizer* opt;

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

    /// @brief Adds a layer to the network
    /// @param layer : Pointer to layer object
    void add_layer(Layer* layer)
    {
      layers.push_back(layer);
    }

    /// @brief Adds a loss function to the network
    /// @param loss_in : Pointer to loss object
    void add_loss(Loss* loss_in)
    {
      loss = loss_in;
    }

    /// @brief Adds an optimizer to the network
    /// @param opt_in : Pointer to optimizer object
    void add_optimizer(Optimizer* opt_in)
    {
      opt = opt_in;
    }

    /// @brief prints a summary of the network
    void summary()
    {
      for (int i = 0; i < layers.size(); i++)
      {
        std::cout << layers[i]->get_layer_type() << " "
                  << "Output shape: " << layers[i]->output_dim() << std::endl;
      }
    };

    // Wrapper functions for forward, backward and update
    // TODO: add fit or train function to handle epochs and batches

    void forward(const Matrix& input);

    /// @brief Using X input and y target, back propagates the loss gradient
    /// @param X
    /// @param y
    void backward(const Matrix& X, const Matrix& y);

    /// @brief
    /// @param opt : Optimizer object reference
    void update(Optimizer& opt);

    /// @brief Forward, backward, and update functions combined for one batch of data
    /// @param X : Input data
    /// @param y : input target
    float train_batch(const Matrix& X, const Matrix& y);

    /// @brief Trains all batches in X and y for one epoch
    /// @param X : Input data
    /// @param y : input target
    /// @param epochs : number of epochs
    /// @param batch_size : size of batches
    /// @param verbose : Level of verbosity
    /// @return
    float train_one_epoch(const Matrix& X, const Matrix& y, const int batch_size, const int verbose);

    /// @brief Fit model to data provided
    /// @param X : Input data
    /// @param y : input target
    /// @param epochs : number of epochs
    /// @param batch_size : size of batches
    /// @param verbose : Level of verbosity
    /// @param shuffle : bool to shuffle data in each epoch
    /// @return
    void fit(
        Matrix& X,  // if const, then cannot shuffle
        Matrix& y,
        const int epochs,
        const int batch_size,
        const int verbose,
        const bool shuffle = true);

    /// @brief Fit model to data provided
    /// @param X : Input data
    /// @return y_pred : Predicted output
    std::vector<int> predict(const Matrix& X);

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

    /// @brief Get the output of a specified layer
    Matrix get_layer_output(int i);

    /// @brief Get the weight matrix of a specified layer
    Matrix get_layer_weight(int i);
  };
}  // namespace nmbuflowtorch
#endif  // NMBUFLOWTORCH_NETWORK_H_
