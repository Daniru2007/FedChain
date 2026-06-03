
package ml;

import math.Matrix;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;

public class NeuralNetwork {
	private final double learningRate;
	private final Matrix[] weights;
	private final Matrix[] biases;
	private final Matrix[] layerOutputs; // includes input at index 0
	private final Matrix[] zs; // pre-activation values for each non-input layer
	private double lastLoss = Double.NaN;

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
		// compute and store loss (MSE) before weight update for tracking
		this.lastLoss = computeMSE(output, target);
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

	/**
	 * Compute and return Mean Squared Error for a single input/target pair.
	 * This runs a forward pass and updates internal activations used by training.
	 */
	public double getLoss(Matrix input, Matrix target) {
		Matrix out = forward(input);
		lastLoss = computeMSE(out, target);
		return lastLoss;
	}

	/**
	 * Returns the most recently computed loss (or NaN if none computed yet).
	 */
	public double getLoss() {
		return lastLoss;
	}

	private static double computeMSE(Matrix a, Matrix b) {
		double[][] da = a.getData();
		double[][] db = b.getData();
		int rows = da.length;
		int cols = rows == 0 ? 0 : da[0].length;
		double sum = 0.0;
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				double d = da[i][j] - db[i][j];
				sum += d * d;
			}
		}
		return sum / Math.max(1, rows * cols);
	}

	/**
	 * Save network weights and biases to a binary file.
	 */
	public void save(String path) throws IOException {
		Path resolved = resolvePersistencePath(path, true);
		try (var out = Files.newOutputStream(resolved)) {
			java.io.DataOutputStream dataOut = new java.io.DataOutputStream(out);
			// write learning rate
			dataOut.writeDouble(learningRate);
			// number of layers (weights length + 1)
			int layers = weights.length + 1;
			dataOut.writeInt(layers);
			// write each weight matrix and bias matrix
			dataOut.writeInt(weights.length);
			for (int i = 0; i < weights.length; i++) {
				double[][] w = weights[i].getData();
				double[][] b = biases[i].getData();
				int wr = w.length;
				int wc = wr == 0 ? 0 : w[0].length;
				dataOut.writeInt(wr);
				dataOut.writeInt(wc);
				for (int r = 0; r < wr; r++) {
					for (int c = 0; c < wc; c++) {
						dataOut.writeDouble(w[r][c]);
					}
				}
				int br = b.length;
				int bc = br == 0 ? 0 : b[0].length;
				dataOut.writeInt(br);
				dataOut.writeInt(bc);
				for (int r = 0; r < br; r++) {
					for (int c = 0; c < bc; c++) {
						dataOut.writeDouble(b[r][c]);
					}
				}
			}
			dataOut.flush();
		}
	}

	/**
	 * Load a network previously saved with {@link #save(String)}.
	 */
	public static NeuralNetwork load(String path) throws IOException {
		Path resolved = resolvePersistencePath(path, false);
		try (var in = Files.newInputStream(resolved)) {
			java.io.DataInputStream dataIn = new java.io.DataInputStream(in);
			double lr = dataIn.readDouble();
			int layers = dataIn.readInt();
			int weightCount = dataIn.readInt();
			Matrix[] weights = new Matrix[weightCount];
			Matrix[] biases = new Matrix[weightCount];
			for (int i = 0; i < weightCount; i++) {
				int wr = dataIn.readInt();
				int wc = dataIn.readInt();
				double[][] w = new double[wr][wc];
				for (int r = 0; r < wr; r++) {
					for (int c = 0; c < wc; c++) {
						w[r][c] = dataIn.readDouble();
					}
				}
				int br = dataIn.readInt();
				int bc = dataIn.readInt();
				double[][] b = new double[br][bc];
				for (int r = 0; r < br; r++) {
					for (int c = 0; c < bc; c++) {
						b[r][c] = dataIn.readDouble();
					}
				}
				weights[i] = new Matrix(w);
				biases[i] = new Matrix(b);
			}
			return new NeuralNetwork(weights, biases, lr);
		}
	}

	private static Path resolvePersistencePath(String path, boolean forSave) {
		Path candidate = Paths.get(path);
		if (candidate.isAbsolute() || candidate.getNameCount() > 1) {
			if (forSave) {
				Path parent = candidate.getParent();
				if (parent != null) {
					try {
						Files.createDirectories(parent);
					} catch (IOException e) {
						throw new RuntimeException("Failed to create persistence directory: " + parent, e);
					}
				}
			}
			return candidate;
		}

		Path modelsDir = Paths.get(System.getProperty("user.dir"), "models");
		if (forSave) {
			try {
				Files.createDirectories(modelsDir);
			} catch (IOException e) {
				throw new RuntimeException("Failed to create models directory: " + modelsDir, e);
			}
		}
		return modelsDir.resolve(candidate);
	}

	// private constructor used by loader
	private NeuralNetwork(Matrix[] weights, Matrix[] biases, double learningRate) {
		this.learningRate = learningRate;
		this.weights = weights;
		this.biases = biases;
		int layers = weights.length + 1;
		this.layerOutputs = new Matrix[layers];
		this.zs = new Matrix[weights.length];
	}
}
