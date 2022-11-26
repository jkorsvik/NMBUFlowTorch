#ifndef NMBUFLOWTORCH_LAYER_H_
#define NMBUFLOWTORCH_LAYER_H_

#include <vector>

#include "./definitions.hpp"
#include "./optimizer.hpp"

namespace nmbuflowtorch
{

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

    virtual void forward(const Matrix& X) = 0;
    virtual void backward(const Matrix& X, const Matrix& accumulated_gradients) = 0;
    virtual void update(Optimizer& opt)
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

    virtual std::vector<float> get_derivatives() const
    {
      return std::vector<float>();
    }

    virtual std::string get_layer_type() const
    {
      return layer_type;
    };
  };

}  // namespace nmbuflowtorch
#endif  // LAYER_H_
