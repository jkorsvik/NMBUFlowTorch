#include "nmbuflowtorch/network.hpp"

#include "nmbuflowtorch/definitions.hpp"
#include "nmbuflowtorch/math_m.hpp"

// Used for pragma unroll which unroll loops when compiling
// performance improvement
const int UNROLLDEPTH = 10;
namespace nmbuflowtorch
{

  float Network::train_batch(const Matrix& X, const Matrix& y)
  {
    this->forward(X);

    this->backward(this->output(), y);
    this->update(*opt);

    return this->loss->output();
  };

  /// @brief  Forward pass of the network using input X
  /// @param X
  /// @param y
  /// @param batch_size
  /// @param verbose Level of verbosity
  /// @return
  float Network::train_one_epoch(const Matrix& X, const Matrix& y, const int batch_size, const int verbose = 1)
  {
    int n_datapoints = X.rows();
    int n_features = X.cols();
    int n_output_size = y.cols();
    int n_batches = (int)(ceil(float(n_datapoints) / float(batch_size)));

    float loss = 0.;

    if (verbose > 1)
    {
      std::cout << "Processing " << n_batches << " Batches of size " << batch_size << " ..." << std::endl;
      printProgress(0.0, n_batches, verbose);
    }
    // sleep(0.01);
    for (int start_idx = 0; start_idx < n_datapoints; start_idx += batch_size)
    {
      int batch_id = start_idx / batch_size;
      Matrix batch_X = X.block(start_idx, 0, std::min(batch_size, n_datapoints - start_idx), n_features);
      Matrix batch_y = y.block(start_idx, 0, std::min(batch_size, n_datapoints - start_idx), n_output_size);
      loss = this->train_batch(batch_X, batch_y);

      if (verbose > 1)
      {
        printProgress(batch_id, n_batches, verbose);
        std::cout << " Loss " << loss;
      }
      // sleep(0.01);
    }

    epoch++;  // Increment epoch counter to use as timesteps in opt
    return loss;
  };

  void Network::fit(Matrix& X, Matrix& y, const int epochs, const int batch_size, const int verbose, const bool shuffle)
  {
    float loss = 0.;
    // TODO: figure out if to copy x and y or not for shuffling, to keep X and y as const

    if (verbose > 0)
    {
      std::cout << std::endl
                << "Fitting model for " << epochs << " Epochs and with Batches of size " << batch_size << " ..."
                << std::endl;
    }
    for (int epoch = 0; epoch < epochs; epoch++)
    {
      if (shuffle)
      {
        shuffle_data(X, y);
      }
      if (verbose > 1)
      {
        std::cout << std::endl << "Epoch: " << epoch << " Loss: " << loss << std::endl;
      }
      if (verbose > 0 && verbose < 2)
      {
        if (epoch % 100 == 0)
        {
          std::cout << std::endl << "Epoch: " << epoch << " Loss: " << loss << std::endl;
        }
      }

      loss = this->train_one_epoch(X, y, batch_size, verbose);
    }
  };

  std::vector<int> Network::predict(const Matrix& X)
  {
    this->forward(X);
    auto net_output = this->layers.back()->output();

    auto cutoff = net_output.unaryExpr(&binary_cutoff);

    std::vector<int> return_vector;

    for (auto x : cutoff.reshaped())
    {
      return_vector.push_back(x);
    }
    return return_vector;

    // std::vector<int> vec(net_output.data())
    // return .cast<int>();
    //  Kan også være test.unaryExpr([](float net_output) {return net_output > 0.5; })
  }

  void Network::forward(const Matrix& input)
  {
    if (layers.empty())
    {
      return;
    }
    layers[0]->forward(input);
    for (int i = 1; i < layers.size(); i++)
    {
      layers[i]->forward(layers[i - 1]->output());
    }
  }

  void Network::backward(const Matrix& input, const Matrix& target)
  {
    int n_layer = layers.size();  // TODO: burde kanskje være en atributt
    // 0 layer
    if (n_layer <= 0)
    {
      return;
    }
    // Evaluates the loss at the output layer
    loss->eval(input, target);
    // If only one layer, then the loss is the gradient
    if (n_layer == 1)
    {
      layers[0]->backward(input, loss->back_gradient());
      // Stops backpropagation here
      return;
    }
    // >1 layers

    layers[n_layer - 1]->backward(layers[n_layer - 2]->output(), loss->back_gradient());

#pragma unroll(UNROLLDEPTH)
    for (int i = n_layer - 2; i > 0; i--)
    {
      auto last_output = layers[i - 1]->output();
      auto gradient_before = layers[i + 1]->back_gradient();
      layers[i]->backward(last_output, gradient_before);
    }
    layers[0]->backward(input, layers[1]->back_gradient());
  }

  void Network::update(Optimizer& opt)
  {
#pragma unroll(UNROLLDEPTH)
    for (int i = 0; i < layers.size(); i++)
    {
      layers[i]->update(opt, epoch);
    }
  }

  std::vector<std::vector<float>> Network::get_parameters() const
  {
    const int n_layer = layers.size();
    std::vector<std::vector<float>> res;
    res.reserve(n_layer);
    for (int i = 0; i < n_layer; i++)
    {
      res.push_back(layers[i]->get_parameters());
    }
    return res;
  }

  void Network::set_parameters(const std::vector<std::vector<float>>& param)
  {
    const int n_layer = layers.size();
    if (static_cast<int>(param.size()) != n_layer)
    {
      throw std::invalid_argument("Parameter size does not match");
    }
    for (int i = 0; i < n_layer; i++)
    {
      layers[i]->set_parameters(param[i]);
    }
  }

  std::vector<std::vector<float>> Network::get_derivatives() const
  {
    const int n_layer = layers.size();
    std::vector<std::vector<float>> res;
    res.reserve(n_layer);
#pragma unroll(UNROLLDEPTH)
    for (int i = 0; i < n_layer; i++)
    {
      res.push_back(layers[i]->get_derivatives());
    }
    return res;
  }

  Matrix Network::get_layer_output(int i)
  {
    return layers[i]->output();
  };
}  // namespace nmbuflowtorch