
#ifndef SRC_MODELS_AUTOENCODER_H_
#define SRC_MODELS_AUTOENCODER_H_

#include "Model.h"

namespace SANN
{

class Autoencoder : public Model
{
public:

	virtual ~Autoencoder()
	{}

	bool validate_model() override
	{
		return (get_list_of_layers().size() >= 3) &&
			   (get_list_of_layers().front()->get_layer_size() == get_list_of_layers().back()->get_layer_size());
	}

	Autoencoder(std::vector<layer_size_t> model_by_layers_size, //layer sizes
							std::vector<act_t> activation_layers, //activation type includes input and output (None is an option)
							std::vector<std::shared_ptr<ANN::Weights>> weights_vec, // Shared pointer to weights type
							double learning_rate = DEFAULT_LEARNING_RATE,
							Optimizers::opt_t optimizer = Optimizers::OPT_ADAM,
							LossFunctionPtr loss_func = std::make_shared<LossFunctions::MSELoss>())
							: Model(model_by_layers_size,activation_layers,
									weights_vec,learning_rate,optimizer,loss_func)
	{}

	Autoencoder(std::vector<uint32_t> model_by_layers_size,
						double learning_rate = DEFAULT_LEARNING_RATE,
						LossFunctionPtr loss_func = std::make_shared<LossFunctions::MSELoss>()) :
					Model(model_by_layers_size,learning_rate,loss_func)
	{}

	Autoencoder(std::list<std::shared_ptr<ANN::Layer>> layers,//layers with their weights
						  double learning_rate = DEFAULT_LEARNING_RATE,
						  LossFunctionPtr loss_func = std::make_shared<LossFunctions::MSELoss>()):
							  Model(layers,learning_rate,loss_func)
	{}


	double train(MatrixXd data,bool print_loss = false)
	{
		MatrixXd labels = data;
		return Model::train(data,labels,print_loss);
	}

private:
	using Model::train;

};

}

#endif /* SRC_MODELS_AUTOENCODER_H_ */