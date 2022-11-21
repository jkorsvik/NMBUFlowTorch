#pragma once

#include "../Models/Model.h"
#include "../Models/ModelLoader.h"
#include "../Models/Autoencoder.h"

#define EXAMPLE_JSON_MODEL "example_model.json"

//using namespace SANN;


int _default_test()
{
	MatrixXd data_mat(4,8); data_mat << 1,2,3,2,3,2,1,0,
										1,2,3,2,3,2,1,0,
										1,2,3,2,3,2,1,0,
										1,2,3,2,3,2,1,0;
//										1,2,3,8,3,2,1,0,
//										1,2,3,8,3,2,1,0,
//										1,2,3,8,3,2,1,0,
//										1,2,3,8,3,2,1,0,
//										1,2,3,8,3,2,1,0,
//										1,2,3,8,3,2,1,0;
	MatrixXd label_mat(4,2); label_mat << 0,1,
										  0,1,
										  0,1,
										  0,1;
//										  0,1,
//										  0,1,
//										  0,1,
//										  0,1,
//										  0,1,
//										  0,1;
	MatrixXd data_with_noise = data_mat;
	data_with_noise += ExtMath::randn(4,8);

	std::vector<uint32_t> layers_sizes{8,4,3,2};
	std::vector<act_t> act_types_vec{act_t::ACT_NONE,act_t::ACT_SIGMOID,act_t::ACT_SIGMOID,act_t::ACT_NONE};
	SANN::Model model(layers_sizes,0.01);
	model.set_activations(act_types_vec);
	model.set_optimizer(Optimizers::OPT_ADAM);//The default is Adam optimizer but you can select another
	model.train(data_mat,label_mat,true);

	MatrixXd results; model.predict(data_with_noise,results);
	std::cout<<"data with noise: \n"<<data_with_noise<<std::endl;
	std::cout<<"results: \n"<<results<<std::endl;

	std::shared_ptr<SANN::Autoencoder> loaded_model;
	SANN::Autoencoder::load_model_from_file(EXAMPLE_JSON_MODEL,loaded_model);

	std::cout<<"Loaded model: "<<loaded_model->get_list_of_layers().front()->get_layer_size()<<std::endl;

	std::cout<<"Done"<<std::endl;

	return 0;
}