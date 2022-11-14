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

// TODO: 
// Treningsloop i Network, regne ut accuracy, Legge til bias i forward pass i dense layer (X*W + b)



class Network
{

    public:
        Network() {

        };

        void fit(){};

        void train_batch(){}


        // Privat?
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

        // https://stats.stackexchange.com/questions/370723/how-to-calculate-the-derivative-of-crossentropy-error-function
        MatrixXd gradient(MatrixXd y, MatrixXd p) {
            MatrixXd ls = - (y.array() / p.array());
            MatrixXd rs = (1-y.array()) / (1 - p.array());

            return ls + rs;
        };
        
};


class Optimizer
// Stochastic Gradient Descent, learning rate satt til 0.001. Burde kanskje flyttes inn i

{
    MatrixXd weight_update;
    bool weight_init = false;

    public:

        Optimizer(){

        };
        
        MatrixXd update(MatrixXd weights, MatrixXd grads_wrt_w) {

            if (weight_init == false) {
                weight_update = MatrixXd::Zero(weights.rows(), weights.cols());
                weight_init = true;
            }
            
            return weights - 0.001 * grads_wrt_w;
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

        Sigmoid(){

        };

        MatrixXd forward(MatrixXd X) {
            return 1 / (1 + (-1 * X).array().exp());
        };

        MatrixXd backward(MatrixXd X) {
            // Sigmoid(x) * (1 - sigmoid(x))

            //left side og right side
            auto ls = forward(X);
            
            auto rs = (1 - ls.array()).matrix();

            // cwiseProduct for elementvis operasjon (ikke vanlig matrisemultiplikasjon)
            return ls.cwiseProduct(rs);;
        };
};


class Dense : public Layer
{
    public:

        int units;
        int input_shape;

        MatrixXd weights;
        MatrixXd bias;

        Optimizer weight_optimizer;
        Optimizer bias_optimizer;

        MatrixXd layer_input;

        Dense(int n_units, int inp_shape) {
            units = n_units;
            input_shape = inp_shape;

            weights = MatrixXd::Random(input_shape, units); // TODO: annen initialisering? Tror det er -1 til 1 her
            bias = MatrixXd::Zero(1, units);

            weight_optimizer = Optimizer();
            bias_optimizer = Optimizer();

        };

        void set_weights(MatrixXd w) {
            // Metode for å overskrive vektene
            weights = w;
        }

        MatrixXd forward(MatrixXd X) {
            layer_input = X; // Holder på input til backward passet
            return  X * weights; // TODO: Plusse på bias
        };

        MatrixXd backward(MatrixXd accumulated_gradients) {

            MatrixXd prev_weights = weights;

            MatrixXd grad_weights = layer_input.transpose() * accumulated_gradients;
            MatrixXd grad_bias = accumulated_gradients.colwise().sum();

            // Oppdaterer vektene
            weights = weight_optimizer.update(weights, grad_weights);
            bias = bias_optimizer.update(bias, grad_bias);

            accumulated_gradients = accumulated_gradients * prev_weights.transpose();

            return accumulated_gradients;
        };

        


};

//toStrings som overskriver << i en stream, burde være mulig å flytte inn i klassene skulle man tro

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