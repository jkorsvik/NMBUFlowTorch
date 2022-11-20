
#ifndef SRC_INCLUDE_MODELLOADER_H_
#define SRC_INCLUDE_MODELLOADER_H_

#include <string>
#include <unordered_map>
#include <fstream>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include "nmbuflowtorch/Layer.h"
#include "nmbuflowtorch/activation_functions.h"

using namespace boost::property_tree;

namespace SANN
{

class ModelLoader
{

public:

	struct model_data_t_
	{
		std::vector<layer_size_t> layer_sizes;
		std::vector<act_t> activation_layers;
		std::vector<std::shared_ptr<ANN::Weights>> weights_vec;
	};

	ModelLoader();
	virtual ~ModelLoader();
	void generate_model_data_from_file(std::string file_path,model_data_t_ &model_data);

private:

	void generate_weights_vec(ptree &pt,model_data_t_ &mdt);
	void load_file(std::string &file_path,model_data_t_ &mdt);
};

}

#endif /* SRC_INCLUDE_MODELLOADER_H_ */