# Neural Network planning document

### Tensor operations
We will be using [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) for all operations in tensors such as multiplication, reshape and more.

##  Project structure

### Layer class
Superclass containing all methods and attributes for a neural network layer.

Methods:
- updateWeights
- forward
- backward

Inheriting layers:

#### Input
Layer containing all logic for the first layer of a model

#### DenseLayer
Operations done in a dense neural network layer
#### Sigmoid
Sigmoid activation funcions

#### Linear
Linear activation function

### Model class
Superclass for all types of models

Methods:
- forward
- backward
- add_layer
- predict
- train (might be moved out to a seperate trainer class later)

Inheriting layers:

#### DenseNeuralNetwork
Model containing Input layer, dense layers, activation functions


# API example

```cpp
DenseNeuralNetwork model = DenseNeuralNetwork();
model.add_layer(Input(256));
model.add_layer(DenseLayer(128));
model.add_layer(Sigmoid());
model.add_layer(DenseLayer(64));
model.add_layer(Sigmoid());

model.train(...);

predictions = model.predict(...);
```
