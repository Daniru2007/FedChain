package ml;

import math.Matrix;

public final class ActivationFunction {
    private ActivationFunction() {
    }

    public static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public static double sigmoidDerivative(double x) {
        double sigmoid = sigmoid(x);
        return sigmoid * (1.0 - sigmoid);
    }

    public static double relu(double x) {
        return Math.max(0.0, x);
    }

    public static double reluDerivative(double x) {
        return x > 0.0 ? 1.0 : 0.0;
    }

    /**
     * Applies softmax to a column vector Matrix (rows x 1).
     * Uses the numerically stable form: exp(x_i - max) / sum(exp(x_j - max)).
     */
    public static Matrix softmax(Matrix z) {
        int rows = z.getRows();
        // find max for numerical stability
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < rows; i++) {
            double v = z.get(i, 0);
            if (v > max) max = v;
        }
        double[] exps = new double[rows];
        double sum = 0.0;
        for (int i = 0; i < rows; i++) {
            exps[i] = Math.exp(z.get(i, 0) - max);
            sum += exps[i];
        }
        double[][] result = new double[rows][1];
        for (int i = 0; i < rows; i++) {
            result[i][0] = exps[i] / sum;
        }
        return new Matrix(result);
    }
}
