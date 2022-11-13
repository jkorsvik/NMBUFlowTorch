#include "nmbuflowtorch/tmp.hpp"
#include "Eigen/Dense"
#include <iostream>
#include <ostream>
#include <cmath>


using namespace std;
using namespace tmp;
using Eigen::MatrixXd;
using Eigen::VectorXf;

#include <Eigen/Dense>
#include <Eigen/Core>

#include <vector>


class Network
{

    public:
        Network() {

        };


        MatrixXd forward(MatrixXd X) {
            

        };

};


class CrossEntropy
{
    public:
        CrossEntropy() {

        };

        MatrixXd loss( MatrixXd y, MatrixXd p) {

            MatrixXd ls = - (y.cwiseProduct(p.array().log().matrix()));

            MatrixXd rs = (1-y.array()).matrix().cwiseProduct((1-p.array()).log().matrix());

            return ls - rs;
            
        };

        
        MatrixXd gradient(MatrixXd y, MatrixXd p) {
            MatrixXd ls = - (y.array() / p.array());
            MatrixXd rs = (1-y.array()) / (1 - p.array());

            return ls + rs;
        };
        
};


class GradientDescent


{
    float learning_rate = 0.01;
    MatrixXd weight_update;
    bool weight_init = false;


    public:

        GradientDescent(){

        };
        
        MatrixXd update(MatrixXd weights, MatrixXd grads_wrt_w) {

            if (weight_init == false) {
                weight_update = MatrixXd::Zero(weights.rows(), weights.cols());
                weight_init = true;
            }
            
            return weights - learning_rate * grads_wrt_w;
            return weights;
        };
        
        
};

class Layer 
{
    public:
        MatrixXd forward(MatrixXd X){};

        MatrixXd backward(MatrixXd X){};


};

class Sigmoid : public Layer
{
    public:
        MatrixXd forward(MatrixXd X) {
            return 1 / (1 + (-1 * X).array().exp());
        };

        MatrixXd backward(MatrixXd X) {
            auto ls = forward(X);
            
            auto rs = (1 - ls.array()).matrix();

            auto res = ls.cwiseProduct(rs);

            return res;
        };
};


class Dense : public Layer
{
    public:

        int units;
        int input_shape;

        MatrixXd weights;
        MatrixXd bias;

        GradientDescent weight_opt;
        GradientDescent bias_opt;

        MatrixXd layer_input;



        Dense(int n_units, int inp_shape) {
            units = n_units;
            input_shape = inp_shape;

            initialize();
        };

        void set_weights(MatrixXd w) {
            weights = w;
        }

        // Kan være privat
        void initialize() {
            
            weights = MatrixXd::Random(input_shape, units); 
            bias = MatrixXd::Zero(1, units);

            weight_opt = GradientDescent();
            bias_opt = GradientDescent();

        };

        MatrixXd forward(MatrixXd X) {
            //cout << X.rows() << X.cols() << endl;
            //cout << weights.rows() << weights.cols() << endl;
            layer_input = X; // Holder på input til backward passet
            MatrixXd res = X * weights; // TODO: Plusse på bias

            return res;

            
        };

        MatrixXd backward(MatrixXd accumulated_grad) {

            MatrixXd prev_weights = weights;

            MatrixXd grad_weights = layer_input.transpose() * accumulated_grad;

            MatrixXd grad_bias = accumulated_grad.colwise().sum();

            // Oppdaterer weights
            weights = weight_opt.update(weights, grad_weights);
            bias = bias_opt.update(bias, grad_bias);

            accumulated_grad = accumulated_grad * prev_weights.transpose();

            return accumulated_grad;
        };

        


};

//toStrings, burde være mulig å flytte inn i klassene

ostream& operator<< (ostream& stream, Layer& obj) {
            return cout << "Basis Layer";
        };

ostream& operator<< (ostream& stream, Dense& obj) {
            return cout << "Dense Layer";
        };

ostream& operator<< (ostream& stream, Sigmoid& obj) {
            return cout << "Sigmoid layer";
        };


int main(int argc, char **argv) {

    // Sammenligner med utregninger fra https://theneuralblog.com/forward-pass-backpropagation-example/
    
    Dense d(2, 2);
    d.initialize();

    MatrixXd W = MatrixXd(2, 2);
    W << 0.1, 0.2, 0.3, 0.4;
    d.set_weights(W);

    CrossEntropy loss_function = CrossEntropy();

    Sigmoid s = Sigmoid();



    MatrixXd y = MatrixXd(2, 2);
    y << 0.05, 0.95, 0.05, 0.95;

    MatrixXd X = MatrixXd(2, 2);
    X << 0.1, 0.5, 0.1, 0.5;


    auto output = d.forward(X);
    cout << output << endl;

    auto output_sig = s.forward(output); 
    //cout << output_sig << endl;

    auto back_sig = s.backward(output);
    //cout << back_sig << endl;


    auto loss = loss_function.loss(y, output);
    //cout << loss << endl;

    auto loss_grad = loss_function.gradient(y, output);
   // cout << loss_grad << endl;

    d.backward(loss_grad);
    

}