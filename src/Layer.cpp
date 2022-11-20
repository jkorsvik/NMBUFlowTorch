#include "nmbuflowtorch/Layer.h"

namespace ANN
{
//************* Layer ************//

//-------- static functions ---------

// connects the layers 
bool Layer::connect_layers(std::weak_ptr<Layer> current_layer,std::weak_ptr<Layer> next_layer,bool random_init)
{

	std::shared_ptr<Layer> next_layer_inst = next_layer.lock();
	std::shared_ptr<Layer> current_layer_inst = current_layer.lock();

	if(next_layer_inst && current_layer_inst)
	{
		next_layer_inst->set_previous_layer_ptr(current_layer);
		current_layer_inst->set_next_layer_ptr(next_layer);

		// Creating an instance of weights
		// rows ---> Next layer
		// cols ---> Current Layer
		// Wx+b where W is NxC and x is Cx1 (N-Next layer dimension,C-Current layer dimension)
		std::shared_ptr<Weights> weights_ptr = std::make_shared<Weights>(next_layer_inst->get_layer_size(),
																		 current_layer_inst->get_layer_size(),
																		 random_init);

		next_layer_inst->set_input_weights(weights_ptr);//new weights are the input of next layer
		current_layer_inst->set_output_weights(weights_ptr);//new weights are the output weights of this layer

		return true;
	}
	return false;
}


bool Layer::connect_layers(std::weak_ptr<Layer> current_layer,std::weak_ptr<Layer> next_layer,std::shared_ptr<Weights> weights_ptr)
{
	std::shared_ptr<Layer> next_layer_inst = next_layer.lock();
	std::shared_ptr<Layer> current_layer_inst = current_layer.lock();

	if(next_layer_inst && current_layer_inst)
	{
		bool weight_mat_size_correct = (weights_ptr->get_weights_mat_ptr()->rows() == next_layer_inst->get_layer_size()) &&
											weights_ptr->get_weights_mat_ptr()->cols() == current_layer_inst->get_layer_size();

		if(!weight_mat_size_correct)
		{
			throw std::runtime_error("Wrong weights matrix - can't connect layers");
		}

		next_layer_inst->set_previous_layer_ptr(current_layer);
		current_layer_inst->set_next_layer_ptr(next_layer);


		next_layer_inst->set_input_weights(weights_ptr);//new weights are the input of next layer
		current_layer_inst->set_output_weights(weights_ptr);//new weights are the output weights of this layer

		return true;
	}
	return false;

}

//-------- getters ----------//

bool Layer::get_has_next()
{
	std::shared_ptr<Layer> next_layer_inst = this->next_layer_ptr_.lock();
	if(next_layer_inst)
	{
		return true;
	}
	return false;
}

bool Layer::get_has_previous()
{
	std::shared_ptr<Layer> previous_layer_inst = this->previous_layer_ptr_.lock();
	if(previous_layer_inst)
	{
		return true;
	}
	return false;
}

bool Layer::set_new_val_to_neurons(VectorXd &new_neurons_val)
{
	if(this->neurons_.size() == new_neurons_val.size())
	{
		this->neurons_ = new_neurons_val;
		return true;
	}
	return false;
}

//********* InputLayer *************//

InputLayer::InputLayer(uint32_t layer_size,ActivationFunctionPtr activation_func,OptimizerPtr optimizer,std::weak_ptr<Layer> next_layer) :
						Layer(layer_size,activation_func,optimizer,std::weak_ptr<Layer>(),next_layer),layer_type(INPUT_LAYER)
{

}




//********* OutputLayer ***********//
OutputLayer::OutputLayer(uint32_t layer_size,ActivationFunctionPtr activation_func,OptimizerPtr optimizer,std::weak_ptr<Layer> previous_layer) :
						Layer(layer_size,activation_func,optimizer,std::weak_ptr<Layer>(),previous_layer),layer_type(OUTPUT_LAYER)
{

}

}