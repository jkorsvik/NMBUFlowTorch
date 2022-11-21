
#ifndef SRC_INCLUDE_MODEL_H_
#define SRC_INCLUDE_MODEL_H_

#include <list>
#include <string>
#include <vector>

#include "../propagation.h"
#include "ModelLoader.h"

#define DEFAULT_OPTIMIZER_OUTPUT_SET -1

namespace SANN
{
	/**
	 * An abstract model class
	 */
	class Model
	{
	protected:

	    double lr_;
	    std::list<std::shared_ptr<ANN::Layer>> layers_;
	    LossFunctionPtr loss_func_;

	    ANN::Propagation::ForwardPropagation fp_;
	    ANN::Propagation::BackwardPropagation bp_;

	    bool layers_connected_;

	    bool connect_layers();

	    void generate_act_vec(std::vector<Activations::act_t> hidden_activations,
							 std::vector<ActivationFunctionPtr> &act_ptr_vec_out,
							 act_t input = ACT_NONE, act_t output = ACT_NONE);

	    bool generate_layers_from_weights(std::vector<std::shared_ptr<ANN::Weights>> weights_vec,
	       							      std::vector<ActivationFunctionPtr> &act_ptr_vec,
	   								      Optimizers::opt_t optimizer = Optimizers::OPT_ADAM);

	public:

	    Model(std::vector<uint32_t> model_by_layers_size,
	    		double learning_rate = DEFAULT_LEARNING_RATE,
				LossFunctionPtr loss_func = std::make_shared<LossFunctions::MSELoss>());

	    Model(std::vector<layer_size_t> model_by_layers_size, //layer sizes
				std::vector<act_t> activation_layers, //activation type includes input and output (None is an option)
				std::vector<std::shared_ptr<ANN::Weights>> weights_vec, // Shared pointer to weights type
				double learning_rate = DEFAULT_LEARNING_RATE,
				Optimizers::opt_t optimizer = Optimizers::OPT_ADAM,
				LossFunctionPtr loss_func = std::make_shared<LossFunctions::MSELoss>());

		Model(double learning_rate = DEFAULT_LEARNING_RATE,LossFunctionPtr loss_func = std::make_shared<LossFunctions::MSELoss>()) :
			lr_(learning_rate),loss_func_(loss_func),fp_(),bp_(learning_rate,loss_func),layers_connected_(false)
	    {}

		Model(std::list<std::shared_ptr<ANN::Layer>> layers,//layers with their weights
			  double learning_rate = DEFAULT_LEARNING_RATE,
			  LossFunctionPtr loss_func = std::make_shared<LossFunctions::MSELoss>()) :
			  lr_(learning_rate),
			  layers_(layers),
			  loss_func_(loss_func),
			  fp_(),
			  bp_(learning_rate,loss_func),
			  layers_connected_(false)
		{}
		virtual ~Model() {};

		virtual bool validate_model() { return layers_.size() >= 2; } //at least two layers to create a valid model

		std::list<std::shared_ptr<ANN::Layer>> get_list_of_layers() { return this->layers_; }

		virtual double train(const MatrixXd &data,const MatrixXd &labels,bool print_loss = false); //return value of final loss
	    void predict(const MatrixXd &data,MatrixXd &y_pred); //returns the predictions - each row is a single prediction

		//setters
	    void set_layers(std::vector<uint32_t> model_by_layers_size,std::vector<Activations::act_t> model_activations = std::vector<Activations::act_t>());
	    void set_activations(std::vector<Activations::act_t> model_activations = std::vector<Activations::act_t>());
	    void set_activations_hidden_only(std::vector<Activations::act_t> model_activations);

		void set_learning_rate(double learning_rate) {this->lr_ = learning_rate; }
		void set_optimizer(Optimizers::opt_t opt_val,int layer_idx = DEFAULT_OPTIMIZER_OUTPUT_SET);
		void set_weights(std::vector <std::shared_ptr<ANN::Weights>> & weights);


		void get_weights(std::vector<std::shared_ptr<ANN::Weights>> &weights);
		void get_neurons(std::vector<VectorXd> &vec_of_layers_of_neurons);
		std::vector<act_t> get_activations_types();
		inline double get_lr() { return this->lr_; }
		inline LossFunctions::loss_t get_loss() { return this->loss_func_->loss_type(); }

		void save_model_to_file();//TODO

		template<class M>
		static void load_model_from_file(std::string model_json_file,std::shared_ptr<M> &model,double lr = DEFAULT_LEARNING_RATE)
		{
			ModelLoader mloader;
			ModelLoader::model_data_t_ mdt;
			mloader.generate_model_data_from_file(model_json_file,mdt);
			model = std::make_shared<M>(mdt.layer_sizes,mdt.activation_layers,mdt.weights_vec);
		}

	};

}



#endif /* SRC_INCLUDE_MODEL_H_ */