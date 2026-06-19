
package ml;

import math.Matrix;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Objects;
import java.util.Random;

public class NeuralNetwork {
	private final double learningRate;
	private final Matrix[] weights;
	private final Matrix[] biases;
	private final Matrix[] layerOutputs; // includes input at index 0
	private final Matrix[] zs; // pre-activation values for each non-input layer
	private double lastLoss = Double.NaN;

	public NeuralNetwork(int[] layerSizes, double learningRate) {
		this(layerSizes, learningRate, new Random());
	}

	public NeuralNetwork(int[] layerSizes, double learningRate, long seed) {
		this(layerSizes, learningRate, new Random(seed));
	}

	private NeuralNetwork(int[] layerSizes, double learningRate, Random rnd) {
		if (layerSizes == null || layerSizes.length < 2) {
			throw new IllegalArgumentException("There must be at least 2 layers (input and output).");
		}
		this.learningRate = learningRate;
		int layers = layerSizes.length;
		this.weights = new Matrix[layers - 1];
		this.biases = new Matrix[layers - 1];
		this.layerOutputs = new Matrix[layers];
		this.zs = new Matrix[layers - 1];

		for (int i = 0; i < layers - 1; i++) {
			int rows = layerSizes[i + 1];
			int cols = layerSizes[i];
			// He initialization: scale = sqrt(2 / fan_in), keeps variance stable through ReLU layers
			double scale = Math.sqrt(2.0 / cols);
			double[][] w = new double[rows][cols];
			double[][] b = new double[rows][1]; // biases start at zero
			for (int r = 0; r < rows; r++) {
				for (int c = 0; c < cols; c++) {
					w[r][c] = rnd.nextGaussian() * scale;
				}
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
			// apply softmax to the output layer, ReLU to all hidden layers
			if (l == weights.length - 1) {
				activation = ActivationFunction.softmax(z);
			} else {
				activation = z.applyFunction(ActivationFunction::relu);
			}
			layerOutputs[l + 1] = activation;
		}
		return activation;
	}

	public void train(Matrix input, Matrix target) {
		// forward pass
		Matrix output = forward(input);
		this.lastLoss = computeCrossEntropy(output, target);
		int L = weights.length;

		// ── Phase 1: compute all gradients using the ORIGINAL weights ────────
		// Output-layer delta: softmax + cross-entropy simplifies to (ŷ - y)
		Matrix delta = output.subtract(target);
		Matrix[] gradWeights = new Matrix[L];
		Matrix[] gradBiases  = new Matrix[L];

		gradWeights[L - 1] = delta.multiply(layerOutputs[L - 1].transpose());
		gradBiases[L - 1]  = delta;

		// Hidden layers — propagate using the original (not yet updated) weights
		for (int l = L - 2; l >= 0; l--) {
			Matrix sp = zs[l].applyFunction(ActivationFunction::reluDerivative);
			delta = weights[l + 1].transpose().multiply(delta).elementWiseMultiply(sp);
			gradWeights[l] = delta.multiply(layerOutputs[l].transpose());
			gradBiases[l]  = delta;
		}

		// ── Phase 2: apply all updates ────────────────────────────────────────
		for (int l = 0; l < L; l++) {
			weights[l] = weights[l].subtract(gradWeights[l].scale(learningRate));
			biases[l]  = biases[l].subtract(gradBiases[l].scale(learningRate));
		}
	}

	public Matrix predict(Matrix input) {
		return forward(input);
	}

	// Re-randomize weights and biases using He initialization
	public void randomizeWeights() {
		Random rnd = new Random();
		for (int i = 0; i < weights.length; i++) {
			int r = weights[i].getRows();
			int c = weights[i].getCols();
			double scale = Math.sqrt(2.0 / c);
			double[][] w = new double[r][c];
			double[][] b = new double[r][1];
			for (int rr = 0; rr < r; rr++) {
				for (int cc = 0; cc < c; cc++) {
					w[rr][cc] = rnd.nextGaussian() * scale;
				}
			}
			weights[i] = new Matrix(w);
			biases[i] = new Matrix(b);
		}
	}

	/**
	 * Compute and return categorical cross-entropy loss for a single input/target pair.
	 * This runs a forward pass and updates internal activations used by training.
	 */
	public double getLoss(Matrix input, Matrix target) {
		Matrix out = forward(input);
		lastLoss = computeCrossEntropy(out, target);
		return lastLoss;
	}

	/**
	 * Returns the most recently computed loss (or NaN if none computed yet).
	 */
	public double getLoss() {
		return lastLoss;
	}

	public Matrix[] getParameters() {
		Matrix[] params = new Matrix[weights.length + biases.length];
		System.arraycopy(weights, 0, params, 0, weights.length);
		System.arraycopy(biases, 0, params, weights.length, biases.length);
		return params;
	}

	public void setParameters(Matrix[] parameters) {
		Objects.requireNonNull(parameters, "Parameters cannot be null.");
		if (parameters.length != weights.length + biases.length) {
			throw new IllegalArgumentException("Parameter count does not match network architecture.");
		}

		for (int i = 0; i < weights.length; i++) {
			validateShape(parameters[i], weights[i].getRows(), weights[i].getCols(), "weight " + i);
		}
		for (int i = 0; i < biases.length; i++) {
			validateShape(parameters[weights.length + i], biases[i].getRows(), biases[i].getCols(), "bias " + i);
		}

		for (int i = 0; i < weights.length; i++) {
			weights[i] = parameters[i];
		}
		for (int i = 0; i < biases.length; i++) {
			biases[i] = parameters[weights.length + i];
		}
	}

	public NeuralNetwork copy() {
		return new NeuralNetwork(copyMatrices(weights), copyMatrices(biases), learningRate);
	}

	/**
	 * Categorical cross-entropy: L = -sum(y_i * log(ŷ_i)).
	 * {@code a} is the softmax output, {@code b} is the one-hot target.
	 * Clamps predictions to [1e-15, 1] to guard against log(0).
	 */
	private static double computeCrossEntropy(Matrix a, Matrix b) {
		double[][] da = a.getData();
		double[][] db = b.getData();
		int rows = da.length;
		double loss = 0.0;
		for (int i = 0; i < rows; i++) {
			double pred = Math.max(da[i][0], 1e-15);
			loss -= db[i][0] * Math.log(pred);
		}
		return loss;
	}

	private static Matrix[] copyMatrices(Matrix[] source) {
		Matrix[] copy = new Matrix[source.length];
		System.arraycopy(source, 0, copy, 0, source.length);
		return copy;
	}

	private static void validateShape(Matrix matrix, int rows, int cols, String name) {
		Objects.requireNonNull(matrix, name + " cannot be null.");
		if (matrix.getRows() != rows || matrix.getCols() != cols) {
			throw new IllegalArgumentException(name + " shape mismatch.");
		}
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
