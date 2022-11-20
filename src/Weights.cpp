#include "nmbuflowtorch/Weights.h"

namespace ANN
{

namespace WeightsNormalization
{
	/**
	 * Normalizing the weights by the standard deviation of weigths_in matrix
	 */
	void NormalizedByInputSize(MatrixXd &weights_in)
	{
		double normalization_factor = sqrt((double)weights_in.cols());
		ExtMath::change_vals_by_func(weights_in,[normalization_factor](double cell_val)
				{
					return cell_val/normalization_factor;
				});
	}
}

/**
 * weights mat default value is 1
 * Bias value default is 0
 */
Weights::Weights(uint32_t rows, uint32_t cols,double weights_val,double bias_val)
{
	this->weights_mat_= weights_val*MatrixXd::Ones(rows,cols);
	this->bias_= bias_val*VectorXd::Ones(rows);
}

Weights::Weights(uint32_t rows, uint32_t cols,bool random_init,double mu, double sig)
{
	if(random_init)
	{
		this->weights_mat_= ExtMath::randn(rows,cols,mu,sig);
		WeightsNormalization::NormalizedByInputSize(this->weights_mat_);
		this->bias_ =  VectorXd::Zero(rows);
	}
	else
	{
		this->weights_mat_= mu*MatrixXd::Ones(rows,cols);
		this->bias_ = VectorXd::Zero(rows);
	}
}


}