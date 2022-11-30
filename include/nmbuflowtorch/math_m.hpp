#ifndef NMBUFLOWTORCH_MATH_M_H_
#define NMBUFLOWTORCH_MATH_M_H_

#include <vector>

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
  // If parallelization is needed, use Eigen::internal::parallel_for
  //
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

// Returns 0 if value is < 0.5, else 1
inline int binary_cutoff(float inp)
{
  return inp >= 0.5;
}

inline float accuracy_score(std::vector<int> y_true, std::vector<int> y_pred)
{
  if (y_true.size() != y_pred.size())
  {
    throw std::runtime_error("Y_true and y_pred are not the same size");
  }
  int correct = 0;
  for (int i = 0; i < y_true.size(); i++)
  {
    if ((int)y_true[i] == (int)y_pred[i])
    {
      correct++;
    }
  }

  return float(correct) / y_true.size();  // Cast to float to avoid integer division
}

#endif  // NMBUFLOWTORCH_MATH_M_H_