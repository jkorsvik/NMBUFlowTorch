#ifndef NMBUFLOWTORCH_NETWORK_H_
#define NMBUFLOWTORCH_NETWORK_H_

#include <stdlib.h>

#include <vector>

#include "./layer.hpp"
#include "./loss.hpp"
#include "./optimizer.hpp"
#include "./utils.h"

namespace nmbuflowtorch
{

  class Network
  {
   private:
    std::vector<Layer*> layers;  // layer pointers
    Loss* loss;                  // loss pointer

    void train_batch(){};

   public:
    Network(Loss loss = NULL){};
    ~Network()
    {
      for (int i = 0; i < layers.size(); i++)
      {
        delete layers[i];
      }
      if (loss)
      {
        delete loss;
      }
    }
    void add_layer(Layer* layer)
    {
      layers.push_back(layer);
    }

    // Private?
    Eigen::MatrixXd forward(Eigen::MatrixXd X){};

    void backward(Eigen::MatrixXd X, Eigen::MatrixXd y){};

    void update(Optimizer& opt){};

    void fit(){};
  };
}  // namespace nmbuflowtorch
#endif  // NMBUFLOWTORCH_NETWORK_H_
