#ifndef NMBUFLOWTORCH_MATH_M_H_
#define NMBUFLOWTORCH_MATH_M_H_

#include <"./definitions.hpp">

inline Eigen::VectorXd dot(Eigen::MatrixXd &W, Eigen::VectorXd &x)
{
  if (W.cols() != x.rows())
  {
    throw std::runtime_error("[ExtMath] Error with given x and W dimensions for dot product");
  }
  return (W * x.asDiagonal()).rowwise().sum();
}

#endif  // NMBUFLOWTORCH_MATH_M_H_