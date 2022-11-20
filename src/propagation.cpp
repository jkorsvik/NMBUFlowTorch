#include "nmbuflowtorch/propagation.h"
#include <iostream>
#define DEBUG_FLAG_FP 0
#define DEBUG_FLAG_BP 0

namespace ANN
{

namespace Propagation
{


/************************************/
/* 		ForwardPropagation 			*/
/************************************/



bool ForwardPropagation::execute()
{
	if(this->input_layer_ptr_)
	{
	std::shared_ptr<Layer> current_layer;
	std::shared_ptr<Layer> next_layer;

	current_layer = this->input_layer_ptr_;

#if DEBUG_FLAG_FP
	int i=1;
#endif

	while ((current_layer->get_layer_type() != OUTPUT_LAYER) && current_layer->get_has_next())
	{

#if DEBUG_FLAG_FP
		std::cout<<"Layer type: "<<current_layer->get_layer_type()<<std::endl;
		std::cout<<"Layer: "<<i++<<std::endl;
#endif
		next_layer = current_layer->get_next_layer_ptr().lock();
		if(next_layer)
		{
			std::shared_ptr<ANN::Weights> output_weights_ptr = current_layer->get_output_weights_ptr();

#if DEBUG_FLAG_FP
			std::cout<<"Weights: \n"<<*output_weights_ptr->get_weights_mat_ptr()<<std::endl;
			std::cout<<"Neurons: \n"<<*current_layer->get_neurons_ptr()<<std::endl;
			std::cout<<"Bias: \n"<<*output_weights_ptr->get_bias_ptr()<<std::endl;
#endif


			VectorXd Wx_val = output_weights_ptr->dot(*current_layer->get_neurons_ptr());

#if DEBUG_FLAG_FP
			std::cout<<"Dot result Wx: \n"<<Wx_val<<std::endl;
#endif

			Wx_val = Wx_val + (*output_weights_ptr->get_bias_ptr());

#if DEBUG_FLAG_FP
			std::cout<<"Result with bias Wx+b: \n"<<Wx_val<<std::endl;
#endif

			VectorXd Wx_val_act = Wx_val.unaryExpr(next_layer->get_activation_func_ptr()->get_func());

#if DEBUG_FLAG_FP
			std::cout<<"f(Wx+b): \n"<<Wx_val_act<<std::endl;
			std::cout<<"===================================\n"<<std::endl;
#endif

			next_layer->set_new_val_to_neurons(Wx_val_act);

		}
		else
		{
			throw std::runtime_error("Forward propagation couldn't lock next layer!");
		}
		current_layer = next_layer;
	}

		return true;
	}
	return false; //input layer doesn't exist
}

/************************************/
/* 		BackwardPropagation 		*/
/************************************/

bool BackwardPropagation::execute(VectorXd &Y)
{
	if(this->output_layer_ptr_)
	{
#if DEBUG_FLAG_BP
		std::cout<<"--------------------------\nBackward Prop\n--------------------------"<<std::endl;
#endif
		std::shared_ptr<Layer> current_layer;
		std::shared_ptr<Layer> previous_layer;

		current_layer = this->output_layer_ptr_;

		error_ = this->loss_func_->func(*(current_layer->get_neurons_ptr()),Y).sum();

#if DEBUG_FLAG_BP
			std::cout<<"error: "<<error_<<std::endl;
#endif
		VectorXd dEtot_dout = this->loss_func_->derivative(*(current_layer->get_neurons_ptr()),Y);//delta between total error to neuron's out

		int l = 0;

		while((current_layer->get_layer_type() != INPUT_LAYER) && current_layer->get_has_previous())
		{
#if DEBUG_FLAG_BP
				std::cout<<"Layer: "<<l<<"\n==========================\n";
#endif
			//dout_dnet - the delta between the output of neurons and its activation function (the derivative of activation function)
			VectorXd dout_dnet = current_layer->get_neurons_ptr()->unaryExpr(current_layer->get_activation_func_ptr()->get_Dfunc());

			previous_layer = current_layer->get_previous_layer_ptr().lock();

#if DEBUG_FLAG_BP
			std::cout<<"current neurons:\n"<<*(current_layer->get_neurons_ptr())<<std::endl;
			std::cout<<"dEtot_dout:\n"<<dEtot_dout<<std::endl;
			std::cout<<"dout_dnet:\n"<<dout_dnet<<std::endl;
#endif



			if(previous_layer)
			{
#if DEBUG_FLAG_BP
				std::cout<<"Previous Neurons: \n"<<*(previous_layer->get_neurons_ptr())<<std::endl;
#endif
				//initialize the size of gradient (grad value per each weight)
				MatrixXd weights_grad(current_layer->get_input_weights_ptr()->get_weights_mat_ptr()->rows(),
									  current_layer->get_input_weights_ptr()->get_weights_mat_ptr()->cols());


				VectorXd bias_diff = VectorXd::Zero(dout_dnet.size());

				for (uint32_t row = 0; row < weights_grad.rows(); row++)
				{
					double etot_dout_sc = dEtot_dout(row);
					double dout_dnet_sc = dout_dnet(row);
					weights_grad.row(row) = *(previous_layer->get_neurons_ptr())*etot_dout_sc*dout_dnet_sc;
					bias_diff(row) = etot_dout_sc*dout_dnet_sc;//bias diff from neurons calculation of delta
				}

				//calculating the delta of Eout with respect to out of hidden
				//dEo/dOutHidden = Etot_dout*dout_dnet*dnet_doutHidden
				//dnet/doutHidden = the attribute of weight which W

				VectorXd Etot_dout_times_dout_dnet = dEtot_dout.array()*dout_dnet.array();

				//dnet_douth - delta between network of current weights and hidden of former is actually the weights
				dEtot_dout = current_layer->get_input_weights_ptr()->get_weights_mat_ptr()->transpose()*Etot_dout_times_dout_dnet;

				current_layer->get_optimizer()->optimize(*(current_layer->get_input_weights_ptr()->get_weights_mat_ptr()),weights_grad,
																		 *(current_layer->get_input_weights_ptr()->get_bias_ptr()),bias_diff,lr_);

#if DEBUG_FLAG_BP
				std::cout<<"dEtot_dout\n"<<dEtot_dout<<std::endl;
				std::cout<<"weights diff: \n"<<weights_grad<<std::endl;
				std::cout<<"weights_mat: \n"<<*(current_layer->get_input_weights_ptr()->get_weights_mat_ptr())<<std::endl;
				std::cout<<"weights_mat after optimize: \n"<<*(current_layer->get_input_weights_ptr()->get_weights_mat_ptr())<<std::endl;

#endif
				current_layer = previous_layer;

				l++;
			}
		}
		return true;
	}
	return false; //output layer doesn't exist
}


}

}