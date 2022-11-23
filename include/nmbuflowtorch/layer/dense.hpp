#ifndef NMBUFLOWTORCH_LAYER_FULLY_CONNECTED_H_
#define NMBUFLOWTORCH_LAYER_FULLY_CONNECTED_H_

#include <vector>

#include "../Layer.hpp"
#include "../Optimizer.hpp"
#include "Eigen/Dense"

namespace nmbuflowtorch::layer
{
  class Dense : public Layer
  {
   private:
    const int input_shape;
    const int units;

    Eigen::MatrixXd weights;
    Eigen::MatrixXd bias;
    Eigen::MatrixXd grad_weights;
    Eigen::MatrixXd grad_bias;

    Optimizer weight_optimizer;
    Optimizer bias_optimizer;

    void set_weight_optimizer(Optimizer& opt)
    {
      weight_optimizer = opt;
    }
    void set_bias_optimizer(Optimizer& opt)
    {
      bias_optimizer = opt;
    }

    void init();

   public:
    Dense(const int input_shape, const int units) : input_shape(input_shape), units(units)
    {
      init();
    }

    void set_optimizer(Optimizer& opt)
    {
      set_weight_optimizer(opt);
      set_bias_optimizer(opt);
    }

    Eigen::MatrixXd forward(const Eigen::MatrixXd& X);
    // TODO : Add backward function with const reference argument
    Eigen::MatrixXd backward(Eigen::MatrixXd& accumulated_gradients);
    // void update(Optimizer& opt);
    int output_dim()
    {
      return units;
    }

    void set_weights(Eigen::MatrixXd new_weights)
    {
      // Metode for å overskrive vektene
      weights = new_weights;
    }

    void set_bias(Eigen::MatrixXd new_bias)
    {
      bias = new_bias;
    };
    Eigen::MatrixXd get_weigths()
    {
      return weights;
    }
    Eigen::MatrixXd get_bias()
    {
      return bias;
    }
    // std::vector<float> get_parameters() const;
    // std::vector<float> get_derivatives() const;
    // void set_parameters(const std::vector<float>& param);
  };
}  // namespace nmbuflowtorch::layer
#endif  // NMBUFLOWTORCH_LAYER_FULLY_CONNECTED_H_
