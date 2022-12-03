#include "nmbuflowtorch/optimizer/adam.hpp"

namespace nmbuflowtorch::optimizer
{
  void Adam::update(Vector::AlignedMapType& w, Vector::ConstAlignedMapType& dw)
  {
    Vector& m = m_map[dw.data()];  // moment map to pointers
    Vector& v = v_map[dw.data()];  // velocity map to pointers

    // m(t) = beta1 * m(t-1) + (1 - beta1) * g(t)
    m = beta1 * m + (1 - beta1) * dw;
    // v(t) = beta2 * v(t-1) + (1 - beta2) * g(t)^2
    v = beta2 * v + (1 - beta2) * dw.cwiseProduct(dw);

    // mhat(t) = m(t) / (1 - beta1(t))
    Vector m_hat = m / (1 - pow(beta1, t + 1));  // m_hat
    // vhat(t) = v(t) / (1 - beta2(t))
    Vector v_hat = v / (1 - pow(beta2, t + 1));  // v_hat

    w -= learning_rate * m_hat / (v_hat.array().sqrt() + epsilon);

    // Update time step
    t += 1;
  }
}  // namespace nmbuflowtorch::optimizer