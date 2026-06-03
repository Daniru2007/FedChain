import math.Matrix;
import ml.NeuralNetwork;

public class Main {
	public static void main(String[] args) {
		int[] layers = {2, 4, 1};
		NeuralNetwork nn = new NeuralNetwork(layers, 0.5, 42L);

		Matrix[] inputs = {
				new Matrix(new double[][]{{0.0}, {0.0}}),
				new Matrix(new double[][]{{0.0}, {1.0}}),
				new Matrix(new double[][]{{1.0}, {0.0}}),
				new Matrix(new double[][]{{1.0}, {1.0}})
		};
		Matrix[] targets = {
				new Matrix(new double[][]{{0.0}}),
				new Matrix(new double[][]{{1.0}}),
				new Matrix(new double[][]{{1.0}}),
				new Matrix(new double[][]{{0.0}})
		};

		int maxEpochs = 10_000;
		double threshold = 0.02;
		double loss = Double.POSITIVE_INFINITY;
		int epoch = 0;

		while (epoch < maxEpochs && loss > threshold) {
			double sum = 0.0;
			for (int i = 0; i < inputs.length; i++) {
				nn.train(inputs[i], targets[i]);
				sum += nn.getLoss();
			}
			loss = sum / inputs.length;
			epoch++;
		}

		System.out.printf("XOR demo finished in %d epochs, average MSE=%.6f%n", epoch, loss);
		for (int i = 0; i < inputs.length; i++) {
			double prediction = nn.predict(inputs[i]).get(0, 0);
			System.out.printf("input=(%.0f, %.0f) -> %.6f%n",
					inputs[i].get(0, 0),
					inputs[i].get(1, 0),
					prediction);
		}
	}
}
