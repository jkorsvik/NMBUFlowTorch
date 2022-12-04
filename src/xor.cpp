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

  int input_size = 2; // Number of columns in training data 
  int output_size = 1; // Number of outputs. Set to 1 for binary problems

  // Defining Eigen matrix for storing input data
  Matrix xor_X = Matrix(4, input_size);
  Matrix xor_y = Matrix(4, 1);

  CSVFormat format;
  format.delimiter(',').no_header();

  CSVReader reader("data/xor/xor.csv", format);

  // The label is the last value in each row in the data. the n_cols < input size switch below is to make sure the last value in the row gets sent to the y matrix
  int n_rows = 0;
  int n_cols = 0;
  for (CSVRow& row : reader)
  {  // Input iterator
    int n_cols = 0;
    for (CSVField& field : row)
    {
      
      if (n_cols < input_size)
      {
        xor_X(n_rows, n_cols) = stof(field.get());  // stof converts string to float
      }
      else
      {
        xor_y(n_rows, n_cols - input_size) = stof(field.get());  // stof converts string to float
      }
      n_cols++;
    }
    n_rows++;
  }

    

  // Initializing network
  nmbuflowtorch::Network xor_net;
  
  // Initializing optimizer (Stochastic gradient descent)
  nmbuflowtorch::optimizer::SGD* xor_opt = new nmbuflowtorch::optimizer::SGD(0.1);


  // Initializing loss (Mean square Error)
  nmbuflowtorch::Loss* xor_loss = new nmbuflowtorch::loss::MSE();

  // Adding optimizer and loss function to the network
  xor_net.add_loss(xor_loss);
  xor_net.add_optimizer(xor_opt);
    

  // Creating layers. Network has the dimension 2 (input) -> 8 -> 1
  nmbuflowtorch::layer::Dense* xor_dense1 = new nmbuflowtorch::layer::Dense(input_size, 8);
  nmbuflowtorch::layer::Sigmoid* xor_sigmoid1 = new nmbuflowtorch::layer::Sigmoid();
  nmbuflowtorch::layer::Dense* xor_dense2 = new nmbuflowtorch::layer::Dense(xor_dense1->output_dim(), 1);
  nmbuflowtorch::layer::Sigmoid* xor_sigmoid2 = new nmbuflowtorch::layer::Sigmoid();

  //Adding layers to network
  xor_net.add_layer(xor_dense1);
  xor_net.add_layer(xor_sigmoid1);
  xor_net.add_layer(xor_dense2);
  xor_net.add_layer(xor_sigmoid2);

  // Training network on xor data
  xor_net.fit(xor_X, xor_y, 10000, 1, 0);

  // Getting binary predictions and converting to vector
  auto xor_y_pred = xor_net.predict(xor_X);
  vector<int> xor_y_true_vector(xor_y.data(), xor_y.data() + xor_y.rows() * xor_y.cols());

  // Calculation accuracy
  auto xor_score = accuracy_score(xor_y_true_vector, xor_y_pred);
  cout << "XOR accuracy score: " << xor_score << endl;
}