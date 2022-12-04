#include <omp.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>

#include "csv_parser/csv.hpp"
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
//#include "nmbuflowtorch/optimizer/nadam.hpp"
#include "nmbuflowtorch/optimizer/sgd.hpp"

// -> is for pointer objects, while . is for value objects
using namespace std;
// CSV parser
using namespace csv;
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
  int input_size = 128 * 128;
  int output_size = 1;

  // Create network
  nmbuflowtorch::Network autoencoder;
  // define loss
  // nmbuflowtorch::Loss* loss = new nmbuflowtorch::loss::CrossEntropy();

  nmbuflowtorch::optimizer::SGD* opt = new nmbuflowtorch::optimizer::SGD(0.1);

  nmbuflowtorch::Loss* loss = new nmbuflowtorch::loss::MSE();

  autoencoder.add_loss(loss);
  autoencoder.add_optimizer(opt);

  // XOR eksempler
  Matrix X = Matrix(2, input_size);  // is also y
  Matrix y = Matrix(2, 1);

  CSVFormat format;
  format.delimiter(',').no_header();

  CSVReader reader("data/autoencoder/circles_128.csv", format);
  int n_rows = 0;
  int n_cols = 0;
  for (CSVRow& row : reader)
  {  // Input iterator
    int n_cols = 0;
    for (CSVField& field : row)
    {
      if (n_cols < input_size)
      {
        X(n_rows, n_cols) = stof(field.get());  // stof converts string to float
      }
      else
      {
        y(n_rows, n_cols - input_size) = stof(field.get());  // stof converts string to float
      }

      // By default, get<>() produces a std::string.
      // A more efficient get<string_view>() is also available, where the resulting
      // string_view is valid as long as the parent CSVRow is alive
      n_cols++;
    }
    n_rows++;
  }

  cout << X << endl;

  // Create layers
  nmbuflowtorch::layer::Dense* dense1 = new nmbuflowtorch::layer::Dense(input_size, 128);
  nmbuflowtorch::layer::Sigmoid* sigmoid1 = new nmbuflowtorch::layer::Sigmoid();
  nmbuflowtorch::layer::Dense* dense2 = new nmbuflowtorch::layer::Dense(dense1->output_dim(), 64);
  nmbuflowtorch::layer::Sigmoid* sigmoid2 = new nmbuflowtorch::layer::Sigmoid();
  nmbuflowtorch::layer::Dense* dense3_compressionlayer = new nmbuflowtorch::layer::Dense(dense2->output_dim(), 32);
  nmbuflowtorch::layer::Sigmoid* sigmoid3_compressionlayer = new nmbuflowtorch::layer::Sigmoid();
  nmbuflowtorch::layer::Dense* dense4 = new nmbuflowtorch::layer::Dense(dense3_compressionlayer->output_dim(), 64);
  nmbuflowtorch::layer::Sigmoid* sigmoid4 = new nmbuflowtorch::layer::Sigmoid();
  nmbuflowtorch::layer::Dense* dense5 = new nmbuflowtorch::layer::Dense(dense4->output_dim(), 64);
  nmbuflowtorch::layer::Sigmoid* sigmoid5 = new nmbuflowtorch::layer::Sigmoid();
  nmbuflowtorch::layer::Dense* dense6 = new nmbuflowtorch::layer::Dense(dense5->output_dim(), 128);
  nmbuflowtorch::layer::Sigmoid* sigmoid6 = new nmbuflowtorch::layer::Sigmoid();
  nmbuflowtorch::layer::Dense* dense7 = new nmbuflowtorch::layer::Dense(dense6->output_dim(), output_size);
  nmbuflowtorch::layer::Sigmoid* sigmoid7 = new nmbuflowtorch::layer::Sigmoid();
  // nmbuflowtorch::layer::Dense* dense3 = new nmbuflowtorch::layer::Dense(dense2->output_dim(), 1);
  // nmbuflowtorch::layer::Sigmoid* sigmoid3 = new nmbuflowtorch::layer::Sigmoid();

  autoencoder.add_layer(dense1);
  autoencoder.add_layer(sigmoid1);
  autoencoder.add_layer(dense2);
  autoencoder.add_layer(sigmoid2);
  autoencoder.add_layer(dense3_compressionlayer);
  autoencoder.add_layer(sigmoid3_compressionlayer);
  autoencoder.add_layer(dense4);
  autoencoder.add_layer(sigmoid4);
  autoencoder.add_layer(dense5);
  autoencoder.add_layer(sigmoid5);
  autoencoder.add_layer(dense6);
  autoencoder.add_layer(sigmoid6);
  autoencoder.add_layer(dense7);
  autoencoder.add_layer(sigmoid7);

  nmbuflowtorch::Network encoder_net;
  encoder_net.add_loss(loss);
  encoder_net.add_optimizer(opt);
  encoder_net.add_layer(dense1);
  encoder_net.add_layer(sigmoid1);
  encoder_net.add_layer(dense2);
  encoder_net.add_layer(sigmoid2);
  encoder_net.add_layer(dense3_compressionlayer);
  encoder_net.add_layer(sigmoid3_compressionlayer);

  nmbuflowtorch::Network decoder_net;
  decoder_net.add_layer(dense4);
  decoder_net.add_layer(sigmoid4);
  decoder_net.add_layer(dense5);
  decoder_net.add_layer(sigmoid5);
  decoder_net.add_layer(dense6);
  decoder_net.add_layer(sigmoid6);
  decoder_net.add_layer(dense7);
  decoder_net.add_layer(sigmoid7);

  autoencoder.fit(X, X, 100, 2, 1);

  // cout << autoencoder.train_batch(X, y) << endl;

  // auto y_pred = autoencoder.forward(X);
  //  for (auto x : y_pred) {
  //    cout << x << endl;
  //  }

  // vector<int> y_true_vector(y.data(), y.data() + y.rows() * y.cols());
  // cout << endl;

  // cout << "ACC : " << accuracy_score(y_true_vector, y_pred) << endl;
  encoder_net.forward(X);
  cout << endl << "Compressed vector representation of image" << endl << encoder_net.output() << endl;

  autoencoder.delete_net();
  //  de net.d;
  //   Cleaning up
  //   delete dense1;
  //   delete sigmoid1;
  //   delete dense2;
  //   delete sigmoid2;
  //   delete loss;
  //   delete opt;
  //   Could  also use smart pointers instead of manual memory management (RAII)
  //   shared pointers could also be used
  //   std::unique_ptr<layer::dense> dense1 = std::make_unique<layer::dense>(arguments...);
}