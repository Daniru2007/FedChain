# FedChain

A small Java workspace with a reusable `math.Matrix` utility and basic ML activation helpers.

## Matrix Utilities

`math.Matrix` is an immutable linear algebra class supporting:
- Construction from dimensions or 2D array
- `add`, `subtract`, `multiply`, `transpose`, `scale`
- `applyFunction(DoubleUnaryOperator)` for element-wise operations
- `elementWiseMultiply` for Hadamard product (used in backprop)

Defensive copying ensures immutability. All operations return new matrices.

## Activation Functions

`ml.ActivationFunction` provides:
- `sigmoid(double)` — standard logistic activation
- `sigmoidDerivative(double)` — needed for backpropagation
- `relu(double)` — rectified linear unit (optional)

## Neural Network

`ml.NeuralNetwork` implements a feedforward network with backpropagation:

### Constructor
```java
NeuralNetwork(int[] layerSizes, double learningRate)
// Example: new NeuralNetwork(new int[]{2, 4, 1}, 0.1)
// Creates a network: 2 inputs → 4 hidden → 1 output, learning rate 0.1
// Weights and biases initialized with small Gaussian noise (stddev 0.01)
```

### Methods
- `forward(Matrix input)` — forward pass through all layers, returns final output
- `train(Matrix input, Matrix target)` — single example training (MSE loss, sigmoid activation, backprop)
- `predict(Matrix input)` — forward pass only, no weight updates
- `randomizeWeights()` — reinitialize weights and biases

### Input Format
- Inputs must be column vectors: shape (inputSize, 1)
- Targets must match output shape: (outputSize, 1)

### Example
```java
int[] layers = {2, 4, 1};
NeuralNetwork nn = new NeuralNetwork(layers, 0.1);

Matrix input = new Matrix(new double[][]{{0.5}, {0.3}});
Matrix target = new Matrix(new double[][]{{1.0}});

// Train on one example
nn.train(input, target);

// Predict
Matrix output = nn.predict(input);
```

## Run tests

```bash
mvn test
```

## Implementation Notes

- All layers use sigmoid activation.
- Backpropagation uses Mean Squared Error (MSE) cost function.
- Currently supports single-example training; extend for minibatch SGD if needed.
- Random initialization uses Gaussian distribution (stddev 0.01). Consider Xavier/He for deeper networks.

