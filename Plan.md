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

Methods:
- forward
- backward
- add_layer
- predict
- train (might be moved out to a seperate trainer class later)


# API example

The API is inspired by the functional API found in Keras. We also considered mimicing the API found in PyTorch, but we believe the Keras API is easer to implement and we are more familiar with it.

```cpp
DenseNeuralNetwork model = Neural Network();
model.add_layer(Input(256));
model.add_layer(DenseLayer(128));
model.add_layer(Sigmoid());
model.add_layer(DenseLayer(64));
model.add_layer(Sigmoid());

model.train(...);

predictions = model.predict(...);
```

# Concurrency

We plan to implement concurrency by splitting the forward and backward passes in the train loop across processes. For example, with a batch size of 4 and 4 processes, the network will process 4*4=16 datapoints at the same time. After the forward pass, the different processess will sync up and reduce to a loss shared by all processes. The loss will then be backpropagated with the same 4*4 datapoints simultaneously.

We are not sure how this will effect the runtime of our software due to the overhead from the concurrency-operations, but implementing it will still be a good learning experience. There are also other ways of parallelizing neural network training, but we believe this is the most realistic way to implement it due to time and complexity constraints.

# Time complexities
Time complexities are hard to determine as they will depend on the sizes of input, number of hidden layers, hidden layer sizes and number of outputs. However, we can estimate a worst case scenario where the forward pass has a worst case O(n^3) where n = max(sizes of input, number of hidden layers, hidden layer sizes and number of outputs). This is due to the time complexity of the standard matrix multiplication algorithm being O(n^3) (There might be a slightly more efficient algorithm being used in eigen, not sure about this)
