#include "nmbuflowtorch/optimizer/nadam.hpp"

namespace nmbuflowtorch::optimizer
{
  void Nadam::update(Vector::AlignedMapType& w, Vector::ConstAlignedMapType& dw)
  {
    Vector& m = m_map[dw.data()];
    Vector& v = v_map[dw.data()];
    if (m.size() == 0)
    {
      m.resize(dw.size());
      m.setZero();
    }
    if (v.size() == 0)
    {
      v.resize(dw.size());
      v.setZero();
    }

    t++;  // increment timestep
    // Update the parameters using Nesterov momentum and Adam optimization
    m = beta1 * m + (1 - beta1) * dw;
    v = beta2 * v + (1 - beta2) * dw.array().square();
    Vector m_hat = m / (1 - pow(beta1, t));
    Vector v_hat = v / (1 - pow(beta2, t));
    // Avoid using the square root function by using the identity: x / sqrt(y) = sqrt(x) / y
    w -= learning_rate * m.array().sqrt() / (v + epsilon);
  }
}  // namespace nmbuflowtorch::optimizer