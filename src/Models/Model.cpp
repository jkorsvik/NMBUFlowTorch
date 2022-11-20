/*
 * Model.cpp
 *
 *  Created on: Feb 5, 2020
 *      Author: david
 */

#include "nmbuflowtorch/Models/Model.h"

#define DEBUG_TRAIN 0

namespace SANN
{
	//constructor

    Model::Model(std::vector<layer_size_t> model_by_layers_size,double learning_rate,LossFunctionPtr loss_func):
				lr_(learning_rate),
				loss_func_(loss_func),
				fp_(),
				bp_(learning_rate,loss_func)
	{
    	set_layers(model_by_layers_size);
    	connect_layers();
	}

    Model::Model(std::vector<layer_size_t> model_by_layers_size,
    			std::vector<act_t> activation_layers,
				std::vector<std::shared_ptr<ANN::Weights>> weights_vec,
    			double learning_rate,
				Optimizers::opt_t optimizer,
				LossFunctionPtr loss_func):
   				lr_(learning_rate),
   				loss_func_(loss_func),
   				fp_(),
   				bp_(learning_rate,loss_func)
    {
		std::vector<ActivationFunctionPtr> act_ptr_vec;
		generate_act_vec(activation_layers,act_ptr_vec);
		if(!generate_layers_from_weights(weights_vec,act_ptr_vec))//also connects the layers
		{
			std::cout<<"[cppSANN] Model wasn't loaded - Can't connect layers!"<<std::endl;
		}
    }

    /**
     * If model activation isn't given then apply default initialization
     */
    void Model::set_layers(std::vector<uint32_t> model_by_layers_size,std::vector<Activations::act_t> model_activations)
    {
   	    layers_.clear();
		std::vector<ActivationFunctionPtr> act_ptr_vec;
    	if(model_activations.empty()) //default activations
    	{
    		model_activations.assign(model_by_layers_size.size()-2,ACT_LEAKY_RELU); //don't put activations to input and output layers
    		generate_act_vec(model_activations,act_ptr_vec);
    	}
    	else //activations were given
    	{
    		for(uint32_t i=0; i < model_by_layers_size.size(); i++)
			{
				act_ptr_vec.push_back(select_activation(model_activations[i]));
			}
    	}

    	for(uint32_t i=0; i < model_by_layers_size.size(); i++)
		{
			if(i == 0) //input size
			{
				std::shared_ptr<ANN::InputLayer> input_layer = std::make_shared<ANN::InputLayer>(model_by_layers_size[i],
																								 act_ptr_vec[i],
																								 std::make_shared<Optimizers::None>());
				layers_.push_back(input_layer);
			}
			else if(i == model_by_layers_size.size()-1)//last element is the output size
			{
				std::shared_ptr<ANN::OutputLayer> output_layer = std::make_shared<ANN::OutputLayer>(model_by_layers_size[i],
																								    act_ptr_vec[i],
																									std::make_shared<Optimizers::Adam>());

				layers_.push_back(output_layer);
			}
			else
			{
				layers_.push_back(std::make_shared<ANN::Layer>(model_by_layers_size[i],act_ptr_vec[i],std::make_shared<Optimizers::None>()));
			}
		}
    }

    void Model::generate_act_vec(std::vector<Activations::act_t> hidden_activations,
								 std::vector<ActivationFunctionPtr> &act_ptr_vec_out,
								 act_t input, act_t output)
	{
    	act_ptr_vec_out.resize(hidden_activations.size()+2);//+ input and output
    	// Generate activations
		for(uint32_t i=0; i < act_ptr_vec_out.size(); i++)
		{
			if(i == 0) //input size
			{
				act_ptr_vec_out[i] = select_activation(input);
			}
			else if(i == act_ptr_vec_out.size()-1)//last element is the output size
			{
				act_ptr_vec_out[i] = select_activation(output);
			}
			act_ptr_vec_out[i] = select_activation(hidden_activations[i]); // hidden layers
		}
	}

    bool Model::generate_layers_from_weights(std::vector<std::shared_ptr<ANN::Weights>> weights_vec,
    								std::vector<ActivationFunctionPtr> &act_ptr_vec,
									Optimizers::opt_t optimizer)
    {
    	if (act_ptr_vec.size() != (weights_vec.size()+1))
    	{
    		return false;
    	}
    	layers_.clear();

    	uint32_t i=0;
		for(; i < weights_vec.size(); i++)
		{

			if(i == 0) //input size
			{
				std::shared_ptr<ANN::InputLayer> input_layer = std::make_shared<ANN::InputLayer>(weights_vec[i]->weights_cols(),
																								 act_ptr_vec[i],
																								 std::make_shared<Optimizers::None>());
				layers_.push_back(input_layer);
			}
			else //hidden layers
			{
				layers_.push_back(std::make_shared<ANN::Layer>(weights_vec[i]->weights_cols(),act_ptr_vec[i],std::make_shared<Optimizers::None>()));
			}
		}

		std::shared_ptr<ANN::OutputLayer> output_layer = std::make_shared<ANN::OutputLayer>(weights_vec.back()->weights_rows(),
																							act_ptr_vec[i],
																							select_optimizer(optimizer));
		layers_.push_back(output_layer);

		uint32_t w=0;
		for (std::list<std::shared_ptr<ANN::Layer>>::iterator it = layers_.begin() ; it != layers_.end() ; it++)
		{
			if((*it)->get_layer_type() != ANN::OUTPUT_LAYER)
			{
				std::list<std::shared_ptr<ANN::Layer>>::iterator next_it = std::next(it,1);
				(*it)->connect_layers(*it,*next_it,weights_vec[w++]);
			}
		}
		this->layers_connected_ = true;
		return true;

    }


    /**
	 * Assumes that layers exists
	 */
    void Model::set_activations_hidden_only(std::vector<Activations::act_t> model_activations)
    {
    	if(model_activations.size() == (layers_.size()-2))//without input and output
    	{
    		model_activations.insert(model_activations.begin(),Activations::ACT_NONE);
    		model_activations.push_back(Activations::ACT_NONE);
    		set_activations(model_activations);
    	}
    }

    /**
     * Assumes that layers exists
     */
    void Model::set_activations(std::vector<Activations::act_t> model_activations)
    {
    	if(model_activations.size() == layers_.size())
    	{
			std::list<std::shared_ptr<ANN::Layer>>::iterator it = layers_.begin();
			for ( int i=0; it != layers_.end(); it++,i++)
			{
				(*it)->set_activation_func_ptr(select_activation(model_activations[i]));
			}
    	}
    }

	/**
	 * If layer index wasn't given then set the optimizer of the last layer
	 */
	void Model::set_optimizer(Optimizers::opt_t opt_val,int layer_idx)
	{
		if(layer_idx == DEFAULT_OPTIMIZER_OUTPUT_SET)
		{
			layers_.back()->set_optimizer(select_optimizer(opt_val));
		}
		else
		{
			std::list<std::shared_ptr<ANN::Layer>>::iterator it = std::next(layers_.begin(),layer_idx);
			(*it)->set_optimizer(select_optimizer(opt_val));
		}

	};

	void Model::set_weights(std::vector <std::shared_ptr<ANN::Weights>> &weights)
	{
		if(weights.size() == layers_.size() - 1)
		{
			auto weights_iterator = weights.begin();
			for (std::list<std::shared_ptr<ANN::Layer>>::iterator layers_it = layers_.begin() ; layers_it != layers_.end() ; layers_it++)
			{
				if((*layers_it)->get_layer_type() != ANN::OUTPUT_LAYER)
				{
					(*layers_it)->set_output_weights(*weights_iterator); //stores weights instance (with bias in it)					(*(layers_it+1))->set_input_weights()
					(*std::next(layers_it))->set_input_weights(*weights_iterator); // set next layer input weights
				}
			}
		}
		else
		{
			throw std::out_of_range("Weights and layers vectors sizes are not equal!");
		}
	}

	void Model::get_weights(std::vector<std::shared_ptr<ANN::Weights>> &weights)
	{
		for (std::list<std::shared_ptr<ANN::Layer>>::iterator it = layers_.begin() ; it != layers_.end() ; it++)
		{
			if((*it)->get_layer_type() != ANN::OUTPUT_LAYER)
			{
				weights.push_back((*it)->get_output_weights_ptr());
			}
		}
	}

	/**
	 * Returns a vector of neurons vectors copies for each layer in model
	 */
	void Model::get_neurons(std::vector<VectorXd> &vec_of_layers_of_neurons)
	{
		for (std::list<std::shared_ptr<ANN::Layer>>::iterator it = layers_.begin() ; it != layers_.end() ; it++)
		{
			vec_of_layers_of_neurons.push_back((*it)->get_neurons());
		}
	}

	/**
	 * Get activation ids of each layer
	 */
	std::vector<act_t> Model::get_activations_types()
	{
		std::vector<act_t> vec_of_activations;
		for (std::list<std::shared_ptr<ANN::Layer>>::iterator it = layers_.begin() ; it != layers_.end() ; it++)
		{
			vec_of_activations.push_back((*it)->get_activation_func_ptr()->act_type());
		}
		return vec_of_activations;
	}



    bool Model::connect_layers()
    {
    	if (validate_model() && !layers_connected_)
    	{
    		for (std::list<std::shared_ptr<ANN::Layer>>::iterator it = layers_.begin() ; it != layers_.end() ; it++)
    		{
    			if((*it)->get_layer_type() != ANN::OUTPUT_LAYER)
    			{
					std::list<std::shared_ptr<ANN::Layer>>::iterator next_it = std::next(it,1);
					(*it)->connect_layers(*it,*next_it);
    			}
    		}
    		layers_connected_ = true;
    		return true;
    	}
		layers_connected_ = false;
    	return false;
    }


    /**
     * Taking data row by row and labels row by row
     * data row size has to be the same as input size
     * labels row size has to be the same as output size
     */
    double Model::train(const MatrixXd &data,const MatrixXd &labels,bool print_loss)
    {
#if DEBUG_TRAIN
    	std::cout<<"data: \n"<<data<<std::endl;
    	std::cout<<"labels: \n"<<labels<<std::endl;
	std::cout<<"in lr: "<<lr_<<std::endl;
#endif
    	//set learning rate and loss function to backward propagation
		bp_.set_params(lr_,loss_func_);

		//set in and out layers to forward and backward propagation instances
		fp_.set_input_layer(layers_.front());
		bp_.set_output_layer(layers_.back());

    	if(!layers_connected_)
    	{
    		throw "Cannot train - Layers are not connected";
    	}
    	if((data.cols() != layers_.front()->get_layer_size()) ||
    	   (labels.cols() != layers_.back()->get_layer_size()))
    	{
    		throw std::runtime_error("Input data or output labels size isn't fit input or output layer!");
    	}

    	for (uint32_t row = 0; row < data.rows(); row++)
    	{
    		VectorXd input_data = (data.row(row)).transpose();
    		VectorXd label_data = (labels.row(row)).transpose();

#if DEBUG_TRAIN
        	std::cout<<"input_data: \n"<<input_data<<std::endl;
        	std::cout<<"label_data: \n"<<label_data<<std::endl;
#endif

    		layers_.front()->set_new_val_to_neurons(input_data);
#if DEBUG_TRAIN
        	std::cout<<"in neurons: \n"<<*(layers_.front()->get_neurons_ptr())<<std::endl;
        	std::cout<<"out neurons: \n"<<*(layers_.back()->get_neurons_ptr())<<std::endl;
#endif
    		fp_.execute();
    		bp_.execute(label_data);
#if DEBUG_TRAIN
        	std::cout<<"out neurons: \n"<<*(layers_.back()->get_neurons_ptr())<<std::endl;
#endif
    		if(print_loss)
    		{
    			std::cout<<"[cppSANN] iteration: "<<row<<" Loss val: "<<bp_.get_error_val()<<std::endl;
    		}
    	}
    	if(print_loss)
		{
			std::cout<<"[cppSANN] Loss val: "<<bp_.get_error_val()<<std::endl;
		}
    	return bp_.get_error_val();
    }

    void Model::predict(const MatrixXd &data,MatrixXd &y_pred)
    {
    	y_pred = MatrixXd(data.rows(),layers_.back()->get_layer_size());
    	if (layers_.front()->get_layer_size() != data.cols())
    	{
    		throw "Input layer and number of features are not equal!";
    	}
    	if(!layers_connected_)
		{
			throw "Cannot predict - Layers are not connected";
		}
		for (uint32_t row = 0; row < data.rows(); row++)
		{
			VectorXd input_data = (data.row(row)).transpose();
			layers_.front()->set_new_val_to_neurons(input_data);
			fp_.set_input_layer(layers_.front());
			fp_.execute();
			y_pred.row(row) = (*layers_.back()->get_neurons_ptr()).transpose();
		}
    }

}


