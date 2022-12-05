#include "nmbuflowtorch/xor.hpp"

namespace nmbuflowtorch
{
  int xor_train(int argc, char** argv)
  {
    int n_epochs = 10000;
    if (argc > 4 && strcmp(argv[3], "--epochs") == 0)
    {
      n_epochs = stoi(argv[4]);
    }
    else
    {
      cout << "If you want to modify epochs:" << endl;
      cout << "nmbuflowtorch [ parallel | not ] [ xor | autoencoder ] --epochs int" << endl;
    }
    int input_size = 2;   // Number of columns in training data
    int output_size = 1;  // Number of outputs. Set to 1 for binary problems

    // Defining Eigen matrix for storing input data
    Matrix xor_X = Matrix(4, input_size);
    Matrix xor_y = Matrix(4, 1);

    CSVFormat format;
    format.delimiter(',').no_header();

    CSVReader reader("data/xor/xor.csv", format);

    // The label is the last value in each row in the data. the n_cols < input size switch below is to make sure the last
    // value in the row gets sent to the y matrix
    int n_rows = 0;
    int n_cols = 0;
    for (CSVRow& row : reader)
    {  // Input iterator
      int n_cols = 0;
      for (CSVField& field : row)
      {
        if (n_cols < input_size)
        {
          xor_X(n_rows, n_cols) = stof(field.get());  // stof converts string to float
        }
        else
        {
          xor_y(n_rows, n_cols - input_size) = stof(field.get());  // stof converts string to float
        }
        n_cols++;
      }
      n_rows++;
    }

    // Initializing network
    Network xor_net;

    // Initializing optimizer (Stochastic gradient descent)
    optimizer::SGD* xor_opt = new optimizer::SGD(0.1);

    // Initializing loss (Mean square Error)
    Loss* xor_loss = new loss::MSE();

    // Adding optimizer and loss function to the network
    xor_net.add_loss(xor_loss);
    xor_net.add_optimizer(xor_opt);

    // Creating layers. Network has the dimension 2 (input) -> 8 -> 1
    layer::Dense* xor_dense1 = new layer::Dense(input_size, 8);
    layer::Sigmoid* xor_sigmoid1 = new layer::Sigmoid();
    layer::Dense* xor_dense2 = new layer::Dense(xor_dense1->output_dim(), 1);
    layer::Sigmoid* xor_sigmoid2 = new layer::Sigmoid();

    // Adding layers to network
    xor_net.add_layer(xor_dense1);
    xor_net.add_layer(xor_sigmoid1);
    xor_net.add_layer(xor_dense2);
    xor_net.add_layer(xor_sigmoid2);

    // Training network on xor data
    xor_net.fit(xor_X, xor_y, n_epochs, 1, 0);

    // Getting binary predictions and converting to vector
    auto xor_y_pred = xor_net.predict(xor_X);
    vector<int> xor_y_true_vector(xor_y.data(), xor_y.data() + xor_y.rows() * xor_y.cols());

    // Calculation accuracy
    auto xor_score = accuracy_score_vec(xor_y_true_vector, xor_y_pred);
    cout << "XOR accuracy score: " << xor_score << endl;

    xor_net.delete_net();
    return 0;
  }
}  // namespace nmbuflowtorch