#include "nmbuflowtorch/optimizer/adam.hpp"

namespace nmbuflowtorch::optimizer
{
  void Adam::update(Vector::AlignedMapType& w, Vector::ConstAlignedMapType& dw, int epoch)
  {
    Vector& m = m_map[dw.data()];
    Vector& v = v_map[dw.data()];

    // update first and second moment estimates
    m = beta1 * m + (1 - beta1) * dw;
    v = beta2 * v + (1 - beta2) * dw.cwiseProduct(dw);

    // update weights
    w -= learning_rate * m / (v.array().sqrt() + epsilon);
  }
}  // namespace nmbuflowtorch::optimizer