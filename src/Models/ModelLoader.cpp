/*
 * ModelLoader.cpp
 *
 *  Created on: Mar 22, 2020
 *      Author: david
 */

#include "nmbuflowtorch/Models/ModelLoader.h"

namespace SANN
{

ModelLoader::ModelLoader() {
	// TODO Auto-generated constructor stub



}

ModelLoader::~ModelLoader() {
	// TODO Auto-generated destructor stub
}

void ModelLoader::generate_model_data_from_file(std::string file_path,model_data_t_ &model_data)
{
	load_file(file_path,model_data);
}

void ModelLoader::load_file(std::string &file_path,model_data_t_ &mdt)
{
	try
	{
		std::ifstream json_ifs(file_path);
		ptree pt;
		json_parser::read_json(json_ifs, pt); //load json file

		for (auto& elem : pt.get_child("activations"))
		{
			mdt.activation_layers.push_back(Activations::str_to_act_t(elem.second.get_value<std::string>()));
		}

		for (auto& elem : pt.get_child("layers"))
		{
			mdt.layer_sizes.push_back(elem.second.get_value<unsigned int>());
		}

		// ---- generate matrices -----//
		generate_weights_vec(pt,mdt);

	}
	catch(...)
	{
		std::cout<<"[cppSANN] ModelLoader has got an invalid json file!"<<std::endl;
	}
}

/**
 * mdt is the model data type that contains the meta data
 * Result of weights extracted from file are saved to mdt.weights_vec
 */
void ModelLoader::generate_weights_vec(ptree &pt,model_data_t_ &mdt)
{
	std::vector <Eigen::MatrixXd> weights_vec; weights_vec.resize(mdt.layer_sizes.size()-1);
	std::vector <Eigen::VectorXd> bias_vec; bias_vec.resize(mdt.layer_sizes.size()-1);
	int w = 0;
	int b = 0;
	enum {WEIGHTS,BIAS};
	for (auto &layer : pt.get_child("weights"))
	{
		//std::cout<<"Layer: "<<layer.first<<std::endl;
		int case_select = (layer.first.find("weight") != std::string::npos) ? WEIGHTS : BIAS;
		switch (case_select)
		{
		  case WEIGHTS: {weights_vec[w] = Eigen::MatrixXd(mdt.layer_sizes[w+1],mdt.layer_sizes[w]); break;}
		  case BIAS:    {bias_vec[b] = Eigen::VectorXd(mdt.layer_sizes[b+1]); break;}
		}

		int i = 0;
		for (auto &row : layer.second)
		{
			switch (case_select)
			{
				case WEIGHTS:	{
					int j = 0;
					for (auto &cell : row.second)
					{
						weights_vec[w](i,j) = cell.second.get_value<double>();
						j++;
					}
					break;
				}
				case BIAS:     {
					bias_vec[b](i) = row.second.get_value<double>(); // @suppress("Field cannot be resolved")
					break;
				}
			}//end of switch case
			i++;
		}
		switch (case_select)
		{
		case WEIGHTS: {w++; break;}
		case BIAS: {b++; break;}
		}
	}
	//convertion to Weights obj
	mdt.weights_vec.resize(mdt.layer_sizes.size()-1);

	for (unsigned i=0; i < weights_vec.size(); i++)
	{
		mdt.weights_vec[i] = std::make_shared<ANN::Weights>(weights_vec[i],bias_vec[i]);
	}
}

}