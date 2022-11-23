set(layer_headers
  include/nmbuflowtorch/layer/ave_pooling.h
  include/nmbuflowtorch/layer/conv.h
  include/nmbuflowtorch/layer/fully_connected.h
  include/nmbuflowtorch/layer/relu.h
  include/nmbuflowtorch/layer/sigmoid.h
  include/nmbuflowtorch/layer/softmax.h
  include/nmbuflowtorch/layer/max_pooling.h
)

set(layer_sources
  src/nmbuflowtorch/layer/ave_pooling.cc
  src/nmbuflowtorch/layer/conv.cc
  src/nmbuflowtorch/layer/fully_connected.cc
  src/nmbuflowtorch/layer/relu.cc
  src/nmbuflowtorch/layer/sigmoid.cc
  src/nmbuflowtorch/layer/softmax.cc
  src/nmbuflowtorch/layer/max_pooling.cc
)

set(loss_headers
  include/nmbuflowtorch/loss/cross_entropy_loss.h
  include/nmbuflowtorch/loss/mse_loss.h
)

set(loss_sources
  src/nmbuflowtorch/loss/cross_entropy_loss.cc
  src/nmbuflowtorch/loss/mse_loss.cc
)

set(optimizer_headers
  include/nmbuflowtorch/optimizer/sgd.h
)

set(optimizer_sources
  src/nmbuflowtorch/optimizer/sgd.cc
)

set(headers
  include/nmbuflowtorch/tmp.hpp # REMOVE ME
  include/nmbuflowtorch/layer.h
  include/nmbuflowtorch/loss.h
  include/nmbuflowtorch/optimizer.h
  include/nmbuflowtorch/network.h
  include/nmbuflowtorch/mnist.h
  include/nmbuflowtorch/utils.h
  ${loss_headers}
  ${layer_headers}
  ${optimizer_headers}
)


set(sources
  src/tmp.cpp # REMOVE ME
  src/main.cpp
  src/nmbuflowtorch/network.cc
  src/nmbuflowtorch/mnist.cc
  ${layer_sources}
  ${loss_sources}
  ${optimizer_sources}
)

set(exe_sources
		src/main.cpp
		${sources}
)


set(directories
  include/eigen3
)

set(test_sources
  src/tmp_test.cpp
)
