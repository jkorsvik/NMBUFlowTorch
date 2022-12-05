#include "nmbuflowtorch/autoencoder.hpp"
namespace nmbuflowtorch
{
  int autoencoder_train(int argc, char** argv)
  {
    // Sammenligner med utregninger fra https://theneuralblog.com/forward-pass-backpropagation-example/
    int n_epochs = 1000;
    if (argc > 4 && strcmp(argv[3], "--epochs") == 0)
    {
      n_epochs = stoi(argv[4]);
    }
    else
    {
      cout << "If you want to modify epochs:" << endl;
      cout << "nmbuflowtorch [ parallel | not ] [ xor | autoencoder ] --epochs int" << endl;
    }
    int input_size = 128 * 128;
    int output_size = input_size;

    // Create network
    Network autoencoder_net;
    // define loss
    // Loss* loss = new loss::CrossEntropy();

    optimizer::Nadam* opt = new optimizer::Nadam(0.01);

    Loss* loss = new loss::MSE();

    autoencoder_net.add_loss(loss);
    autoencoder_net.add_optimizer(opt);

    // XOR eksempler
    Matrix X = Matrix(2, input_size);  // is also y
    Matrix y = Matrix(2, 1);

    CSVFormat format;
    format.delimiter(',').no_header();

    CSVReader reader("data/autoencoder/circles_128.csv", format);
    int n_rows = 0;
    int n_cols = 0;
    for (CSVRow& row : reader)
    {  // Input iterator
      int n_cols = 0;
      for (CSVField& field : row)
      {
        if (n_cols < input_size)
        {
          X(n_rows, n_cols) = stof(field.get());  // stof converts string to float
        }
        else
        {
          y(n_rows, n_cols - input_size) = stof(field.get());  // stof converts string to float
        }

        // By default, get<>() produces a std::string.
        // A more efficient get<string_view>() is also available, where the resulting
        // string_view is valid as long as the parent CSVRow is alive
        n_cols++;
      }
      n_rows++;
    }

    // Create layers
    layer::Dense* dense1 = new layer::Dense(input_size, 128);
    layer::Sigmoid* sigmoid1 = new layer::Sigmoid();
    layer::Dense* dense2 = new layer::Dense(dense1->output_dim(), 64);
    layer::Sigmoid* sigmoid2 = new layer::Sigmoid();
    layer::Dense* dense3_compressionlayer = new layer::Dense(dense2->output_dim(), 32);
    layer::Sigmoid* sigmoid3_compressionlayer = new layer::Sigmoid();
    layer::Dense* dense4 = new layer::Dense(dense3_compressionlayer->output_dim(), 64);
    layer::Sigmoid* sigmoid4 = new layer::Sigmoid();
    layer::Dense* dense5 = new layer::Dense(dense4->output_dim(), 64);
    layer::Sigmoid* sigmoid5 = new layer::Sigmoid();
    layer::Dense* dense6 = new layer::Dense(dense5->output_dim(), 128);
    layer::Sigmoid* sigmoid6 = new layer::Sigmoid();
    layer::Dense* dense7 = new layer::Dense(dense6->output_dim(), output_size);
    layer::Sigmoid* sigmoid7 = new layer::Sigmoid();
    // layer::Dense* dense3 = new layer::Dense(dense2->output_dim(), 1);
    // layer::Sigmoid* sigmoid3 = new layer::Sigmoid();

    autoencoder_net.add_layer(dense1);
    autoencoder_net.add_layer(sigmoid1);
    autoencoder_net.add_layer(dense2);
    autoencoder_net.add_layer(sigmoid2);
    autoencoder_net.add_layer(dense3_compressionlayer);
    autoencoder_net.add_layer(sigmoid3_compressionlayer);
    autoencoder_net.add_layer(dense4);
    autoencoder_net.add_layer(sigmoid4);
    autoencoder_net.add_layer(dense5);
    autoencoder_net.add_layer(sigmoid5);
    autoencoder_net.add_layer(dense6);
    autoencoder_net.add_layer(sigmoid6);
    autoencoder_net.add_layer(dense7);
    autoencoder_net.add_layer(sigmoid7);

    Network encoder_net;
    encoder_net.add_loss(loss);
    encoder_net.add_optimizer(opt);
    encoder_net.add_layer(dense1);
    encoder_net.add_layer(sigmoid1);
    encoder_net.add_layer(dense2);
    encoder_net.add_layer(sigmoid2);
    encoder_net.add_layer(dense3_compressionlayer);
    encoder_net.add_layer(sigmoid3_compressionlayer);

    Network decoder_net;
    decoder_net.add_layer(dense4);
    decoder_net.add_layer(sigmoid4);
    decoder_net.add_layer(dense5);
    decoder_net.add_layer(sigmoid5);
    decoder_net.add_layer(dense6);
    decoder_net.add_layer(sigmoid6);
    decoder_net.add_layer(dense7);
    decoder_net.add_layer(sigmoid7);

    autoencoder_net.fit(X, X, n_epochs, 2, 2);

    // cout << autoencoder_net.train_batch(X, y) << endl;

    // auto y_pred = autoencoder_net.forward(X);
    //  for (auto x : y_pred) {
    //    cout << x << endl;
    //  }

    // vector<int> y_true_vector(y.data(), y.data() + y.rows() * y.cols());
    // cout << endl;
    encoder_net.forward(X);
    Matrix compressed_M = encoder_net.output();

    cout << endl << "Compressed vector representation of image" << endl << compressed_M << endl;
    cout << endl
         << "Compressed size: " << compressed_M.cols() * compressed_M.rows() * 32 / 1009 << " KB" << endl
         << "Uncompressed size: " << X.cols() * X.rows() * 32 / 1000 << " KB" << endl;

    decoder_net.forward(compressed_M);
    Matrix reconstructed_M = decoder_net.output();

    cout << endl
         << "Reconstructed image size: " << reconstructed_M.cols() * reconstructed_M.rows() * 32 / 1000 << " KB" << endl;

    cout << endl << "Pixel accuracy" << endl << accuracy_score_matrix(X, reconstructed_M) << endl;

    // DELETES ALL POINTERS IN THE NETWORK, SO DON'T USE THEM AFTER THIS
    autoencoder_net.delete_net();
    //  de net.d;
    //   Cleaning up
    //   delete dense1;
    //   delete sigmoid1;
    //   delete dense2;
    //   delete sigmoid2;
    //   delete loss;
    //   delete opt;
    //   Could  also use smart pointers instead of manual memory management (RAII)
    //   shared pointers could also be used
    //   std::unique_ptr<layer::dense> dense1 = std::make_unique<layer::dense>(arguments...);
    return 0;
  }
}  // namespace nmbuflowtorch