#include "nmbuflowtorch/optimizer/nadam.hpp"

namespace nmbuflowtorch::optimizer
{
  void Nadam::update(Vector::AlignedMapType& w, Vector::ConstAlignedMapType& dw, int epoch)
  {
    int t = epoch;
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

    // Update the parameters using Nesterov momentum and Adam optimization
    m = beta1 * m + (1 - beta1) * dw;
    v = beta2 * v + (1 - beta2) * dw.cwiseProduct(dw);
    const Vector m_hat = m / (1 - std::pow(beta1, t));
    const Vector v_hat = v / (1 - std::pow(beta2, t));

    Vector delta = learning_rate * m_hat.array() / (v_hat.array().sqrt() + epsilon);

    w = w - delta;
  }
}  // namespace nmbuflowtorch::optimizer