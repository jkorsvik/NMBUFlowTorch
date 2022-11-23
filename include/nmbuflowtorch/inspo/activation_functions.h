#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <functional>

namespace Activations
{
  typedef enum
  {
    ACT_NONE,
    ACT_IDENTITY,
    ACT_SIGMOID,
    ACT_RELU,
    ACT_LEAKY_RELU,
    ACT_SWISH,
    ACT_ELU,
    ACT_TANH,
    ACT_GAUSSIAN,
    ACT_TOTAL
  } act_t;

  typedef std::function<double(double)> t_activation_func;

  class ActivationFunction
  {
   public:
    virtual double function(double x) = 0;
    virtual double function_derivative(double x) = 0;  // first derivative of function
    virtual act_t act_type() = 0;

    virtual ~ActivationFunction()
    {
    }

    virtual t_activation_func get_func()
    {
      return [this](double x) { return this->function(x); };
    }
    virtual t_activation_func get_Dfunc()
    {
      return [this](double x) { return this->function_derivative(x); };
    }  // first derivative of function
  };

  /**
   * Sigmoid
   * Pros:
          - Squashes numbers to range [0,1]
          - Interpretation as a saturating “firing rate” of a neuron
          Cons:
          - Saturated neurons “kill” the gradients
          - Sigmoid outputs are not zero-centered
          - exp() is a bit compute expensive
   */
  class Sigmoid : public ActivationFunction
  {
   public:
    inline double function(double x)
    {
      return 1. / (1. + std::exp(-x));
    }

    inline double function_derivative(double x)
    {
      double sigmoid_x = function(x);
      return sigmoid_x * (1 - sigmoid_x);
    }

    inline act_t act_type()
    {
      return ACT_SIGMOID;
    }
  };

  /**
   * Swish
   * y = x*Sigmoid(x)
   */
  class Swish : public ActivationFunction
  {
   private:
    double B_;
    Sigmoid sigmoid_;

   public:
    inline double function(double x)
    {
      return B_ * x * sigmoid_.function(x);
    }

    inline double function_derivative(double x)
    {
      double sigmoid_x = sigmoid_.function(x);
      return B_ * (sigmoid_x + x * sigmoid_x * (1 - sigmoid_x));
    }

    inline act_t act_type()
    {
      return ACT_SWISH;
    }

    Swish(double b = 1) : B_(b){};
  };

  /**
   * Rectified Linear Unit
   * Pros:
          - Does not saturate
          - Very computationally efficient
          - Converges much faster than
          - sigmoid/tanh in practice (e.g. 6x)
          - Actually more biologically
          - plausible than sigmoid
          Cons:
          - Not zero-centered output
          - For x>0 - exploding gradient with output [0,inf]
          - Dead neurons when value reaches 0.
          - for x<0 gradient is 0 therefore weights will not get adjusted.
   */
  class ReLU : public ActivationFunction
  {
   public:
    inline double function(double x)
    {
      return std::max(0., x);
    }

    inline double function_derivative(double x)
    {
      return x > 0 ? 1 : 0;
    }

    inline act_t act_type()
    {
      return ACT_RELU;
    }
  };

  class LeakyReLU : public ActivationFunction
  {
    double a_;  // alpha of leaky relue

   public:
    inline double function(double x)
    {
      return std::max(a_ * x, x);
    }

    inline double function_derivative(double x)
    {
      return x > 0 ? 1 : a_;
    }

    inline act_t act_type()
    {
      return ACT_LEAKY_RELU;
    }

    LeakyReLU(double a = 0.001) : a_(a){};
  };

  /**
   * Exponential Linear Units
   * Pros:
          - All benefits of ReLU
          - Closer to zero mean outputs
          - Negative saturation regime compared with leaky ReLU adds some robustness to noise
     Cons:
      - Given x>0, exploding gradient with an output range of [0,inf]

          Note: argument a should be set here or use lambda function to envelope it
   */

  class ELU : public ActivationFunction
  {
    double a_;  // alpha of leaky relue

   public:
    inline double function(double x)
    {
      return x > 0 ? x : a_ * (std::exp(x) - 1);
    }

    inline double function_derivative(double x)
    {
      return x > 0 ? 1 : a_ * std::exp(x);
    }

    inline act_t act_type()
    {
      return ACT_ELU;
    }

    ELU(double a = 0.001) : a_(a){};
  };

  /**
   * Tanh
   *
   * Pros:
   * 		- Squashes numbers to range [-1,1]
   * 		- Zero centered
   * Cons:
   * 		- Kills gradients when saturated.
   *
   */
  class Tanh : public ActivationFunction
  {
   public:
    inline double function(double x)
    {
      return tanh(x);
    }

    inline double function_derivative(double x)
    {
      double r = tanh(x);
      return 1 - r * r;
    }

    inline act_t act_type()
    {
      return ACT_TANH;
    }
  };

  class Gaussian : public ActivationFunction
  {
   public:
    inline double function(double x)
    {
      return std::exp(-x * x);
    }

    inline double function_derivative(double x)
    {
      return -2 * x * std::exp(-x * x);
    }

    inline act_t act_type()
    {
      return ACT_GAUSSIAN;
    }
  };

  /**
   * Doesn't impact anything - Mainly for debug or layers without activation function such as output layer
   */
  class Identity : public ActivationFunction
  {
   public:
    inline double function(double x)
    {
      return x;
    }

    inline double function_derivative(double x)
    {
      return 1.;
    }

    inline act_t act_type()
    {
      return ACT_IDENTITY;
    }
  };

  class None : public Identity
  {
   public:
    using Identity::function;
    using Identity::function_derivative;

    inline act_t act_type()
    {
      return ACT_NONE;
    }
  };

  static std::vector<std::string> str_to_activation_enum = {
    "none", "sigmoid", "relu", "leaky_relu", "swish", "elu", "tanh"
  };

  inline act_t str_to_act_t(std::string act_name)
  {
    unsigned int res = ACT_NONE;
    for (std::string act_str : str_to_activation_enum)
    {
      if (act_str == act_name)
      {
        return act_t(res);
      }
      res++;
    }
    return ACT_NONE;
  }

}  // namespace Activations

typedef std::shared_ptr<Activations::ActivationFunction> ActivationFunctionPtr;

/**
 * Select activation by an enum  from Activations::act_t
 */
inline ActivationFunctionPtr select_activation(Activations::act_t ActVal)
{
  ActivationFunctionPtr chosen_act;
  switch (ActVal)
  {
    case Activations::ACT_NONE:
    {
      chosen_act = std::make_shared<Activations::None>();
      break;
    }
    case Activations::ACT_IDENTITY:
    {
      chosen_act = std::make_shared<Activations::Identity>();
      break;
    }
    case Activations::ACT_SIGMOID:
    {
      chosen_act = std::make_shared<Activations::Sigmoid>();
      break;
    }
    case Activations::ACT_RELU:
    {
      chosen_act = std::make_shared<Activations::ReLU>();
      break;
    }
    case Activations::ACT_LEAKY_RELU:
    {
      chosen_act = std::make_shared<Activations::LeakyReLU>();
      break;
    }
    case Activations::ACT_SWISH:
    {
      chosen_act = std::make_shared<Activations::Swish>();
      break;
    }
    case Activations::ACT_ELU:
    {
      chosen_act = std::make_shared<Activations::ELU>();
      break;
    }
    case Activations::ACT_TANH:
    {
      chosen_act = std::make_shared<Activations::Tanh>();
      break;
    }
    case Activations::ACT_GAUSSIAN:
    {
      chosen_act = std::make_shared<Activations::Gaussian>();
      break;
    }
    default:
    {
      chosen_act = std::make_shared<Activations::None>();
    }
  }

  return chosen_act;
}

#define DEFAULT_ACTIVATION_FUNC std::make_shared<Activations::Sigmoid>()
