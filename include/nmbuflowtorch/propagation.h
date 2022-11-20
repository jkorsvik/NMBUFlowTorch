
#ifndef SRC_INCLUDE_PROPAGATION_H_
#define SRC_INCLUDE_PROPAGATION_H_

#include <memory>
#include "loss_functions.h"
#include "Layer.h"

#define DEFAULT_LEARNING_RATE 1e-4

namespace ANN
{

namespace Propagation
{

class ForwardPropagation
{

private:

	std::shared_ptr<InputLayer> input_layer_ptr_;

public:

	ForwardPropagation() {}

	ForwardPropagation(std::shared_ptr<InputLayer> input_layer_ptr) : input_layer_ptr_(input_layer_ptr)
	{

	}

	void set_input_layer(std::shared_ptr<InputLayer> input_layer_ptr)
	{
		this->input_layer_ptr_ = input_layer_ptr;
	}
	//setters
	void set_input_layer(std::shared_ptr<Layer> input_layer_ptr)
	{
		if (input_layer_ptr->get_layer_type() == ANN::INPUT_LAYER)
		{
			std::shared_ptr<InputLayer> in_layer = std::dynamic_pointer_cast<InputLayer>(input_layer_ptr);
			set_input_layer(in_layer);
		}
	}
	bool execute();


};


/**
 * Backward propagation class
 * The algorithm is based on the excellent explanation given by Matt Mazur:
 * "A Step by Step Backpropagation Example" - https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
 */
class BackwardPropagation
{

private:

	std::shared_ptr<OutputLayer> output_layer_ptr_;
	double lr_;//learning rate
	LossFunctionPtr loss_func_;
	double error_;


public:

	BackwardPropagation(std::shared_ptr<OutputLayer> output_layer_ptr,
						double learning_rate = DEFAULT_LEARNING_RATE,
						LossFunctionPtr loss_func = std::make_shared<LossFunctions::MSELoss>()) :
						output_layer_ptr_(output_layer_ptr),
						lr_(learning_rate),
						loss_func_(loss_func),
						error_(0)
	{}
	BackwardPropagation(double learning_rate = DEFAULT_LEARNING_RATE,
						LossFunctionPtr loss_func = std::make_shared<LossFunctions::MSELoss>()) :
						lr_(learning_rate),
						loss_func_(loss_func),
						error_(0)
	{}

	//setters



	inline void set_output_layer(std::shared_ptr<Layer> output_layer_ptr)
	{
		if (output_layer_ptr->get_layer_type() == ANN::OUTPUT_LAYER)
		{
			std::shared_ptr<OutputLayer> out_layer = std::dynamic_pointer_cast<OutputLayer>(output_layer_ptr);
			set_output_layer(out_layer);
		}
	}

	inline void set_output_layer(std::shared_ptr<OutputLayer> output_layer_ptr)
	{
		this->output_layer_ptr_ = output_layer_ptr;
	}

	inline void set_params(double learning_rate,LossFunctionPtr loss_func)
	{
		this->lr_ = learning_rate;
		this->loss_func_ = loss_func;
	}

	bool execute(VectorXd &Y);

	inline double get_error_val() {return this->error_;}
};

}//end of namespace Propagation

}//end of namespace ANN

#endif /* SRC_INCLUDE_PROPAGATION_H_ */