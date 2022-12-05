#include "nmbuflowtorch/optimizer/sgd.hpp"

namespace nmbuflowtorch::optimizer
{
  void SGD::update(Vector::AlignedMapType& w, Vector::ConstAlignedMapType& dw, int epoch)
  {
    Vector& v = v_map[dw.data()];
    v = dw;
    w -= learning_rate * v;
  }
}  // namespace nmbuflowtorch::optimizer