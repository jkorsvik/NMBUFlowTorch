#ifndef SRC_TESTS_AUTOENCODER_TEST_H_
#define SRC_TESTS_AUTOENCODER_TEST_H_

#include "nmbuflowtorch/Models/Autoencoder.h"

int ae_test()
{
	std::cout<<"Autoencoder test \n---------------------"<<std::endl;
	std::vector<uint32_t> layers_sizes{16,8,4,8,16};
	std::vector<act_t> act_types_vec{act_t::ACT_NONE,act_t::ACT_LEAKY_RELU,act_t::ACT_LEAKY_RELU,act_t::ACT_LEAKY_RELU,act_t::ACT_NONE};
	double lr = 1e-4;
	SANN::Autoencoder model(layers_sizes,lr);
	model.set_optimizer(Optimizers::OPT_ADAM);
	model.set_activations(act_types_vec);

	std::cout<<"loss function: "<<model.get_loss()<<std::endl;

	MatrixXd rand_mat = (MatrixXd::Random(128,16)+MatrixXd::Ones(128,16))/2;
	MatrixXd data_mat(rand_mat.rows()*4,rand_mat.cols());
	data_mat << rand_mat,rand_mat,rand_mat,rand_mat;
	MatrixXd res;
	//std::cout<<"Input matrix: \n"<<data_mat<<std::endl;


	double loss;
	loss = model.train(data_mat);
	std::cout<<"1st training loss: "<<loss<<std::endl;
	loss = model.train(data_mat);
	std::cout<<"2nd training loss: "<<loss<<std::endl;
	loss = model.train(data_mat);
	std::cout<<"3rd training loss: "<<loss<<std::endl;
	loss = model.train(data_mat);
	std::cout<<"4rd training loss: "<<loss<<std::endl;

	model.predict(data_mat,res);
	//std::cout<<"Results: \n"<<res<<std::endl;

	double distance = (res-data_mat).array().abs().sum();

	std::cout<<"distance: "<<distance<<std::endl;


	return 0;
}



#endif /* SRC_TESTS_AUTOENCODER_TEST_H_ */