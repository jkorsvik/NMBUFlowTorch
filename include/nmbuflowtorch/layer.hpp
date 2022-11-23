#ifndef NMBUFLOWTORCH_LAYER_H_
#define NMBUFLOWTORCH_LAYER_H_

#include <Eigen/Dense>
#include <vector>

// #include "./Optimizer.hpp"
// #include "./utils.h"

namespace nmbuflowtorch
{

  class Layer
  {
   protected:
    Eigen::MatrixXd layer_input;
    Eigen::MatrixXd layer_output;

   public:
    virtual ~Layer()
    {
    }

    Eigen::MatrixXd forward(const Eigen::MatrixXd& X);
    Eigen::MatrixXd backward(const Eigen::MatrixXd& accumulated_gradients);
    // TODO: Add update function to layer class and get it out of backward
    // virtual void update(Optimizer& opt)

    virtual const Eigen::MatrixXd& get_last_input()
    {
      return layer_input;
    }

    virtual const Eigen::MatrixXd& get_output()
    {
      return layer_output;
    }

    virtual int output_dim()
    {
      return -1;
    }

    virtual std::vector<float> get_parameters() const
    {
      return std::vector<float>();
    }
    virtual void set_parameters(const std::vector<float>& param)
    {
    }
  };

}  // namespace nmbuflowtorch
#endif  // LAYER_H_
