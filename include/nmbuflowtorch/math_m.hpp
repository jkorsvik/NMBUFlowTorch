#ifndef NMBUFLOWTORCH_MATH_M_H_
#define NMBUFLOWTORCH_MATH_M_H_

#include <cblas.h>

#include "./definitions.hpp"

/// @brief Calculates the dot product of a matrix and a vector.
/// @param W The input matrix.
/// @param x The input vector.
/// @return A vector containing the dot product of the matrix and vector.
inline Vector MVdot(const Matrix& W, const Vector& x)
{
  if (W.cols() != x.rows())
  {
    throw std::runtime_error("[ExtMath] Error with given x and W dimensions for dot product");
  }
  return (W * x.asDiagonal()).rowwise().sum();
}

/// @brief Calculates the dot product of two matrices.
/// @param W The first input matrix.
/// @param X The second input matrix.
/// @return A matrix containing the dot product of the two input matrices.
inline Matrix MMdot(const Matrix& W, const Matrix& X)
{
  if (X.cols() != W.rows())
  {
    throw std::runtime_error("[ExtMath] Error with given x and W dimensions for dot product");
    // MVdot(W, Vector::cast(X))
  }
  // return (W.cwiseProduct(X));
  // If parallelization is needed, use Eigen::internal::parallel_for
  //
  return X * W;
}

/// @brief Finds the maximum value in each column of a matrix and returns the corresponding indices.
/// @param m The input matrix.
/// @return A vector containing the indices of the maximum value in each column of the input matrix.
inline Vector colwise_max_index(Matrix& m)
{
  Vector indices(m.cols());

  for (size_t i = 0; i < m.cols(); i++)
  {
    float current_max_val;
    int index;

    for (size_t j = 0; j < m.rows(); j++)
    {
      if (j == 0 || m(j, i) > current_max_val)
      {
        index = j;
        current_max_val = m(j, i);
      }

      indices(i) = index;
    }
  }

  return indices;
}

/// @brief Applies a binary cutoff to a given float value.
/// @param inp The input float value.
/// @return 0 if the input value is less than 0.5, or 1 if the input value is greater than or equal to 0.5.
inline int binary_cutoff(float inp)
{
  return inp >= 0.5;
}

/// @brief Calculates the accuracy score for a given set of true and predicted labels.
/// @param y_true The true labels.
/// @param y_pred The predicted labels.
/// @return The accuracy score, which is the fraction of true and predicted labels that match.
inline float accuracy_score(std::vector<int> y_true, std::vector<int> y_pred)
{
  if (y_true.size() != y_pred.size())
  {
    throw std::runtime_error("Y_true and y_pred are not the same size");
  }
  int correct = 0;
  for (int i = 0; i < y_true.size(); i++)
  {
    if ((int)y_true[i] == (int)y_pred[i])
    {
      correct++;
    }
  }

  return float(correct) / y_true.size();  // Cast to float to avoid integer division
}

/// @brief Shuffles a dataset by creating a permutation matrix from a given vector of indices and applying it to the data
/// matrices.
/// @param X Matrix of shape (n_samples, n_features) containing the input data.
/// @param y Matrix of shape (n_samples, n_outputs) containing the corresponding targets.
inline void shuffle_data(Matrix& X, Matrix& y)
{
  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(X.rows());

  perm.setIdentity();
  std::random_shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size());
  X = perm * X;
  y = perm * y;
}

// https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/blas-and-sparse-blas-routines/blas-routines/blas-level-3-routines/cblas-gemm.html
// https://github.com/higucheese/cblas_sgemm
// c = a * b
inline void BLAS_mmul_sgeem(
    Matrix& __restrict c,  // IS NOT CONST SINCE IT COUNTAINS BIAS ALREADY AND WILL BE THE RESULTING MATRIX
    const Matrix& __restrict a,
    const Matrix& __restrict b,
    // Tranpose flags
    bool aT = false,
    bool bT = false,
    bool rowmajor = true)
{
  enum CBLAS_TRANSPOSE transA = aT ? CblasTrans : CblasNoTrans;
  enum CBLAS_TRANSPOSE transB = bT ? CblasTrans : CblasNoTrans;
  enum CBLAS_ORDER rowmajorfl = bT ? CblasRowMajor : CblasColMajor;

  size_t M = c.rows();
  size_t N = c.cols();
  size_t K = aT ? a.rows() : a.cols();

  float alpha = 1.0f;
  float beta = 1.0f;

  //  C ← αAB+βC or C ← αABT+βC
  //  C ← αATB+βC or	C ← αATBT+βC

  size_t lda = aT ? K : M;
  size_t ldb = bT ? N : K;
  size_t ldc = M;

  // https://www.ibm.com/docs/en/essl/6.2?topic=reference-basic-linear-algebra-subprograms-blas-blas-cblas
  // double general matrix-matrix multiplication C = alpha*op(A)*op(B) + beta*C
  // Could also be SGEMM
  cblas_sgemm(rowmajorfl, transA, transB, M, N, K, alpha, a.data(), lda, b.data(), ldb, beta, c.data(), ldc);
}

#endif  // NMBUFLOWTORCH_MATH_M_H_