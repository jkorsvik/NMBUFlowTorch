#pragma once

#include <Eigen/Core>
#include <random>
#include <iostream>
#include <cstdint>

namespace ExtMath //extended math operations to eigen
{

inline Eigen::MatrixXd randn(uint32_t rows, uint32_t cols,double mu = 0.,double sig = 1.)
{
	std::default_random_engine generator;
	std::normal_distribution<double> distribution(mu,sig);
	Eigen::MatrixXd newMat(rows,cols);

	for (int row = 0; row < newMat.rows(); row++)
	{
		for (int col=0; col < newMat.cols(); col++)
		{
			newMat(row,col) = distribution(generator);
		}
	}
	return newMat;
}

/**
 * function changes the value of each cell in mat by given func
 */
inline void set_vals_by_func(Eigen::MatrixXd &mat, std::function<double()> &func)
{
	for (int row = 0; row < mat.rows(); row++)
	{
		for (int col=0; col < mat.cols(); col++)
		{
			mat(row,col) = func();
		}
	}
}

/**
 *  function changes the value of each cell in mat by given func. func receives current cell valu as a param.
 */
inline void change_vals_by_func(Eigen::MatrixXd &mat,std::function<double(double)> func)
{
	for (int row = 0; row < mat.rows(); row++)
	{
		for (int col=0; col < mat.cols(); col++)
		{
			mat(row,col) = func(mat(row,col));
		}
	}
}

/**
 * Implementation of W*x dot product
 * W is a matrix mxn
 * x is a column vector nx1
 *
 * return column vector mx1
 */
inline Eigen::VectorXd dot(Eigen::MatrixXd &W, Eigen::VectorXd &x)
{
	if (W.cols() != x.rows())
	{
		throw std::runtime_error("[ExtMath] Error with given x and W dimensions for dot product");
	}
	return (W*x.asDiagonal()).rowwise().sum();
}

/**
 * Calculating mean squared error sum{(Y_est-Y)^2}
 */
inline double mse(Eigen::VectorXd &Y,Eigen::VectorXd &Y_est)
{
	Eigen::VectorXd Y_tot = Y_est-Y;
	return ((Y_tot.array()*Y_tot.array()).sum())/Y.array().size();
}

}