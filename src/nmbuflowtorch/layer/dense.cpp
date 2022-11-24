#include "nmbuflowtorch/layer/dense.hpp"

namespace nmbuflowtorch::layer
{

  void Dense::init()
  {
    weights.resize(input_shape, units);  // TODO: annen initialisering? Tror det er -1 til 1 her
    bias.resize(units);
    weights.setRandom();
    bias.setRandom();

    grad_weights.resize(input_shape, units);
    grad_bias.resize(units);
    grad_weights.setZero();
    grad_bias.setZero();
  }

  void Dense::forward(const Matrix& X)
  {
    layer_input = X;                   // Holder på input til backward passet
    layer_output = MMdot(weights, X);  // TODO: Plusse på bias WX + B
    layer_output.colwise() += bias;    // Eigen::Matrix3Xf::Ones(1,)
    // mtx += Eigen::Matrix3Xf::Ones(3,4);
  }

  void Dense::backward(const Matrix& X, const Matrix& accumulated_gradients)
  {
    grad_weights = layer_input.transpose() * accumulated_gradients;
    grad_bias = accumulated_gradients.colwise().sum();

    // accumulated_gradients = accumulated_gradients * weights.transpose();

    gradient_back.resize(input_shape, layer_input.cols());
    gradient_back = weights * accumulated_gradients;
  }

  void Dense::update(Optimizer& opt)
  {
    Vector::AlignedMapType weights_vec(weights.data(), weights.size());
    Vector::AlignedMapType bias_vec(bias.data(), bias.size());
    Vector::ConstAlignedMapType grad_weights_vec(grad_weights.data(), grad_weights.size());
    Vector::ConstAlignedMapType grad_bias_vec(grad_bias.data(), grad_bias.size());

    opt.update(weights_vec, grad_weights_vec);
    opt.update(bias_vec, grad_bias_vec);

    // opt.update(weights, grad_weights);
    // opt.update(bias, grad_bias);
  }

  /// @brief dense layer parameters
  /// @return vector of parameters, LAST element is bias
  std::vector<float> Dense::get_parameters() const
  {
    std::vector<float> res(weights.size() + bias.size());
    // Copy the data of weights and bias to a long vector
    std::copy(weights.data(), weights.data() + weights.size(), res.begin());
    std::copy(bias.data(), bias.data() + bias.size(), res.begin() + weights.size());
    return res;
  }

  void Dense::set_parameters(const std::vector<float>& param)
  {
    if (static_cast<int>(param.size()) != grad_weights.size() + bias.size())
      throw std::invalid_argument("Parameter size does not match");
    std::copy(param.begin(), param.begin() + weights.size(), weights.data());
    std::copy(param.begin() + bias.size(), param.end(), bias.data());
  }
  /// @brief dense layer derivates
  /// @return vector of derivates, LAST element is bias
  std::vector<float> Dense::get_derivatives() const
  {
    std::vector<float> res(grad_weights.size() + grad_bias.size());
    // Copy the data of weights and bias to a long vector
    std::copy(grad_weights.data(), grad_weights.data() + grad_weights.size(), res.begin());
    std::copy(grad_bias.data(), grad_bias.data() + grad_bias.size(), res.begin() + grad_weights.size());
    return res;
  }

}  // namespace nmbuflowtorch::layer