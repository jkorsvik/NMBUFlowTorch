set(layer_headers
  include/nmbuflowtorch/layer/dense.hpp
  include/nmbuflowtorch/layer/sigmoid.hpp
  include/nmbuflowtorch/layer/relu.hpp
  
)

set(layer_sources
  src/nmbuflowtorch/layer/dense.cpp
  src/nmbuflowtorch/layer/sigmoid.cpp
  src/nmbuflowtorch/layer/relu.cpp
  
)

set(loss_headers
  include/nmbuflowtorch/loss/cross_entropy.hpp
  include/nmbuflowtorch/loss/mse.hpp
)

set(loss_sources
  src/nmbuflowtorch/loss/cross_entropy.cpp
  src/nmbuflowtorch/loss/mse.cpp
)

set(optimizer_headers
  include/nmbuflowtorch/optimizer/sgd.hpp
  #include/nmbuflowtorch/optimizer/adam.hpp
  #include/nmbuflowtorch/optimizer/nadam.hpp
)

set(optimizer_sources
  src/nmbuflowtorch/optimizer/sgd.cpp
  #src/nmbuflowtorch/optimizer/adam.cpp
  #src/nmbuflowtorch/optimizer/nadam.cpp
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
    #src/main_multithread.cpp
    #src/main_single_thread.cpp
    src/main.cpp
		${sources}
)


set(directories
  include/eigen3
)

set(test_sources
  src/main_test.cpp
)
