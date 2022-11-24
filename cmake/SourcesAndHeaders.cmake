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
  include/nmbuflowtorch/definitions.hpp 
  include/nmbuflowtorch/math_m.hpp
  include/nmbuflowtorch/layer.hpp
  include/nmbuflowtorch/loss.hpp
  include/nmbuflowtorch/optimizer.hpp
  include/nmbuflowtorch/network.hpp
  #include/nmbuflowtorch/mnist.h
  #include/nmbuflowtorch/utils.h
  ${loss_headers}
  ${layer_headers}
  ${optimizer_headers}
)


set(sources
  src/nmbuflowtorch/network.cpp
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
  src/main_test.cpp
)
