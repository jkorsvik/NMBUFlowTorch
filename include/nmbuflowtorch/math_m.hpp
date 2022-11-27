#ifndef NMBUFLOWTORCH_MATH_M_H_
#define NMBUFLOWTORCH_MATH_M_H_

#include "./definitions.hpp"

inline Vector MVdot(const Matrix& W, const Vector& x)
{
  if (W.cols() != x.rows())
  {
    throw std::runtime_error("[ExtMath] Error with given x and W dimensions for dot product");
  }
  return (W * x.asDiagonal()).rowwise().sum();
}

inline Matrix MMdot(const Matrix& W, const Matrix& X)
{
  if (X.cols() != W.rows())
  {
    throw std::runtime_error("[ExtMath] Error with given x and W dimensions for dot product");
    // MVdot(W, Vector::cast(X))
  }
  // return (W.cwiseProduct(X));
  return X * W;
}

inline Vector colwise_max_index(Matrix& m)
{
  Vector indices(m.cols());

  for (size_t i = 0; i < m.cols(); i++)
  {
    float current_max_val;
    int index;

    for (size_t j = 0; j < m.rows(); j++)
    {
      if (j == 0 || m(j, i) > current_max_val)
      {
        index = j;
        current_max_val = m(j, i);
      }

      indices(i) = index;
    }
  }

  return indices;
}

#endif  // NMBUFLOWTORCH_MATH_M_H_