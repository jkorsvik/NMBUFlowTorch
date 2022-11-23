set(layer_headers
  include/nmbuflowtorch/layer/dense.hpp
  include/nmbuflowtorch/layer/sigmoid.hpp
  
)

set(layer_sources
  src/nmbuflowtorch/layer/dense.cpp
  src/nmbuflowtorch/layer/sigmoid.cpp
  
)

set(loss_headers
  include/nmbuflowtorch/loss/cross_entropy_loss.hpp
)

set(loss_sources
  src/nmbuflowtorch/loss/cross_entropy_loss.cpp
)

set(optimizer_headers
  include/nmbuflowtorch/optimizer/sgd.hpp
)

set(optimizer_sources
  src/nmbuflowtorch/optimizer/sgd.cpp
)

set(headers
  include/nmbuflowtorch/tmp.hpp # REMOVE ME
  include/nmbuflowtorch/Layer.hpp
  include/nmbuflowtorch/Loss.hpp
  include/nmbuflowtorch/Optimizer.hpp
  #include/nmbuflowtorch/Network.hpp
  #include/nmbuflowtorch/mnist.h
  #include/nmbuflowtorch/utils.h
  ${loss_headers}
  ${layer_headers}
  ${optimizer_headers}
)


set(sources
  src/tmp.cpp # REMOVE ME
  src/main.cpp
  #src/nmbuflowtorch/network.cc
  #src/nmbuflowtorch/mnist.cc
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
