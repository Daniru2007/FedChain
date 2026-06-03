package ml;

import math.Matrix;

import java.util.Random;

public class NeuralNetwork {
	private final double learningRate;
	private final Matrix[] weights;
	private final Matrix[] biases;
	private final Matrix[] layerOutputs; // includes input at index 0
	private final Matrix[] zs; // pre-activation values for each non-input layer

	public NeuralNetwork(int[] layerSizes, double learningRate) {
		if (layerSizes == null || layerSizes.length < 2) {
			throw new IllegalArgumentException("There must be at least 2 layers (input and output).");
		}
		this.learningRate = learningRate;
		int layers = layerSizes.length;
		this.weights = new Matrix[layers - 1];
		this.biases = new Matrix[layers - 1];
		this.layerOutputs = new Matrix[layers];
		this.zs = new Matrix[layers - 1];

		Random rnd = new Random();
		for (int i = 0; i < layers - 1; i++) {
			int rows = layerSizes[i + 1];
			int cols = layerSizes[i];
			double[][] w = new double[rows][cols];
			double[][] b = new double[rows][1];
			// small random initialization
			for (int r = 0; r < rows; r++) {
				for (int c = 0; c < cols; c++) {
					w[r][c] = rnd.nextGaussian() * 0.01;
				}
				b[r][0] = rnd.nextGaussian() * 0.01;
			}
			this.weights[i] = new Matrix(w);
			this.biases[i] = new Matrix(b);
		}
	}

	public Matrix forward(Matrix input) {
		if (input == null) throw new IllegalArgumentException("Input cannot be null.");
		// expect column vector of size inputLayer x 1
		layerOutputs[0] = input;
		Matrix activation = input;
		for (int l = 0; l < weights.length; l++) {
			Matrix z = weights[l].multiply(activation).add(biases[l]);
			zs[l] = z;
			activation = z.applyFunction(ActivationFunction::sigmoid);
			layerOutputs[l + 1] = activation;
		}
		return activation;
	}

	public void train(Matrix input, Matrix target) {
		// forward pass
		Matrix output = forward(input);

		// compute delta for output layer: delta = (a - y) * sigmoid'(z)
		int L = weights.length;
		Matrix delta = output.subtract(target).elementWiseMultiply(zs[L - 1].applyFunction(ActivationFunction::sigmoidDerivative));

		// gradient for weights[L-1] = delta * a_{L-1}^T
		Matrix aPrev = layerOutputs[L - 1];
		Matrix gradW = delta.multiply(aPrev.transpose());
		weights[L - 1] = weights[L - 1].subtract(gradW.scale(learningRate));
		biases[L - 1] = biases[L - 1].subtract(delta.scale(learningRate));

		// backpropagate
		for (int l = L - 2; l >= 0; l--) {
			Matrix wNext = weights[l + 1];
			Matrix z = zs[l];
			Matrix sp = z.applyFunction(ActivationFunction::sigmoidDerivative);
			delta = wNext.transpose().multiply(delta).elementWiseMultiply(sp);

			aPrev = layerOutputs[l];
			gradW = delta.multiply(aPrev.transpose());
			weights[l] = weights[l].subtract(gradW.scale(learningRate));
			biases[l] = biases[l].subtract(delta.scale(learningRate));
		}
	}

	public Matrix predict(Matrix input) {
		return forward(input);
	}

	// Re-randomize weights and biases with small gaussian noise
	public void randomizeWeights() {
		Random rnd = new Random();
		for (int i = 0; i < weights.length; i++) {
			int r = weights[i].getRows();
			int c = weights[i].getCols();
			double[][] w = new double[r][c];
			double[][] b = new double[r][1];
			for (int rr = 0; rr < r; rr++) {
				for (int cc = 0; cc < c; cc++) {
					w[rr][cc] = rnd.nextGaussian() * 0.01;
				}
				b[rr][0] = rnd.nextGaussian() * 0.01;
			}
			weights[i] = new Matrix(w);
			biases[i] = new Matrix(b);
		}
	}
}
