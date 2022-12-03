#include "nmbuflowtorch/optimizer/nadam.hpp"

namespace nmbuflowtorch::optimizer
{
  void Nadam::update(Vector::AlignedMapType& w, Vector::ConstAlignedMapType& dw)
  {
    Vector& m = m_map[dw.data()];  // moment map to pointers
    Vector& v = v_map[dw.data()];  // velocity map to pointers

    m = beta1 * m + (1 - beta1) * dw;
    v = beta2 * v + (1 - beta2) * dw.cwiseProduct(dw);

    Vector m_hat = (1 - pow(beta1, t)) * m / (1 - pow(beta1, t)) + beta1 * m / (1 - pow(beta1, t - 1));
    Vector v_hat = (1 - pow(beta2, t)) * v / (1 - pow(beta2, t)) + beta2 * v / (1 - pow(beta2, t - 1));

    w -= learning_rate * m_hat / (v_hat.array().sqrt() + epsilon);
  }
}  // namespace nmbuflowtorch::optimizer