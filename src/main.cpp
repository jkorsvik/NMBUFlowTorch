#include <cblas.h>
#include <omp.h>

#include "nmbuflowtorch/definitions.hpp"  // imports and renames important files

// Path: src/main.cpp
#include "nmbuflowtorch/autoencoder.hpp"
#include "nmbuflowtorch/xor.hpp"

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

  if (argc > 2 && strcmp(argv[2], "xor") == 0)
  {
    nmbuflowtorch::xor_train(argc, argv);
  }
  else if (argc > 2 && strcmp(argv[2], "autoencoder") == 0)
  {
    nmbuflowtorch::autoencoder_train(argc, argv);
  }
  else
  {
    cout << "Please specify a program to run like this:" << endl;
    cout << "nmbuflowtorch [ parallel | not ] [ xor | autoencoder ]" << endl;
  }
  cout << "PROGRAM FINISHED" << endl;
  return 0;
}