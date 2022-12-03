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

    // Update the parameters using Nesterov momentum and Adam optimization
    m = beta1 * m + (1 - beta1) * dw;
    v = beta2 * v + (1 - beta2) * dw.cwiseProduct(dw);
    Vector momentum = (1 - beta1) * (m / (1 - pow(beta1, t))) + beta1 * (v / (1 - pow(beta2, t)));
    w = w - learning_rate * momentum;
  }
}  // namespace nmbuflowtorch::optimizer