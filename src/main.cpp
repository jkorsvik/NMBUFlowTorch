#include <omp.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>

#include "nmbuflowtorch/definitions.hpp"  // imports and renames important files
#include "nmbuflowtorch/layer.hpp"
#include "nmbuflowtorch/layer/dense.hpp"
#include "nmbuflowtorch/layer/sigmoid.hpp"
#include "nmbuflowtorch/loss.hpp"
#include "nmbuflowtorch/loss/cross_entropy.hpp"
#include "nmbuflowtorch/loss/mse.hpp"
#include "nmbuflowtorch/math_m.hpp"
#include "nmbuflowtorch/network.hpp"
#include "nmbuflowtorch/optimizer.hpp"
//#include "nmbuflowtorch/optimizer/adam.hpp"
#include "nmbuflowtorch/optimizer/nadam.hpp"
#include "nmbuflowtorch/optimizer/sgd.hpp"

// -> is for pointer objects, while . is for value objects
using namespace std;

int main(int argc, char** argv)
{
  if (argc > 1 && strcmp(argv[1], "parallel") == 0)
  {
#define EIGEN_USE_BLAS  // use BLAS for matrix multiplication
// includes to make Eigen use BLAS+LAPACK
#include <complex>

#define EIGEN_SUPERLU_SUPPORT
#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACKE

#define LAPACK_COMPLEX_CUSTOM
#define lapack_complex_float  std::complex<float>
#define lapack_complex_double std::complex<double>
    Eigen::initParallel();
    int n_threads = omp_get_max_threads();
    omp_set_num_threads(n_threads);

    cout << "Running parallely with max cores: " << omp_get_max_threads() << endl;
    n_threads = Eigen::nbThreads();
    // OMP_NUM_THREADS = n./ my_program omp_set_num_threads(n);
    Eigen::setNbThreads(n_threads);
  }
  else
  {
    int single_thread = 1;
#ifdef EIGEN_DONT_PARALLELIZE
    cout << "something";
#endif
    cout << "Running single threaded program" << endl;
    // Run main program
    omp_set_num_threads(single_thread);
    Eigen::setNbThreads(single_thread);
  }

  // Sammenligner med utregninger fra https://theneuralblog.com/forward-pass-backpropagation-example/
  int input_size = 2;
  int n_classes = 1;

  // Create network
  nmbuflowtorch::Network net;
  // define loss
  // nmbuflowtorch::Loss* loss = new nmbuflowtorch::loss::CrossEntropy();

  nmbuflowtorch::optimizer::Nadam* opt = new nmbuflowtorch::optimizer::Nadam(0.1);

  nmbuflowtorch::Loss* loss = new nmbuflowtorch::loss::MSE();

  net.add_loss(loss);
  net.add_optimizer(opt);

  // XOR eksempler
  Matrix X = Matrix(4, 2);
  X << 0, 0, 0, 1, 1, 0, 1, 1;

  Matrix y = Matrix(4, 1);
  y << 0, 1, 1, 0;

  // Create layers
  nmbuflowtorch::layer::Dense* dense1 = new nmbuflowtorch::layer::Dense(input_size, 4);
  nmbuflowtorch::layer::Sigmoid* sigmoid1 = new nmbuflowtorch::layer::Sigmoid();
  nmbuflowtorch::layer::Dense* dense2 = new nmbuflowtorch::layer::Dense(dense1->output_dim(), n_classes);
  nmbuflowtorch::layer::Sigmoid* sigmoid2 = new nmbuflowtorch::layer::Sigmoid();
  // nmbuflowtorch::layer::Dense* dense3 = new nmbuflowtorch::layer::Dense(dense2->output_dim(), 1);
  // nmbuflowtorch::layer::Sigmoid* sigmoid3 = new nmbuflowtorch::layer::Sigmoid();

  net.add_layer(dense1);
  net.add_layer(sigmoid1);
  net.add_layer(dense2);
  net.add_layer(sigmoid2);
  // net.add_layer(dense3);
  // net.add_layer(sigmoid3);

  net.fit(X, y, 100000, 64, 0);

  // cout << net.train_batch(X, y) << endl;

  auto y_pred = net.predict(X);
  // for (auto x : y_pred) {
  //   cout << x << endl;
  // }

  vector<int> y_true_vector(y.data(), y.data() + y.rows() * y.cols());
  cout << endl;

  cout << "ACC : " << accuracy_score(y_true_vector, y_pred) << endl;

  delete net;
  // Cleaning up
  // delete dense1;
  // delete sigmoid1;
  // delete dense2;
  // delete sigmoid2;
  // delete loss;
  // delete opt;
  // Could  also use smart pointers instead of manual memory management (RAII)
  // shared pointers could also be used
  // std::unique_ptr<layer::dense> dense1 = std::make_unique<layer::dense>(arguments...);
}