package ml;

import math.Matrix;

/**
 * Measures classification accuracy of a {@link NeuralNetwork} on a labelled dataset.
 *
 * <p>The network is expected to produce a softmax probability vector (one output
 * per class). Predicted class = argmax of the output. Ground-truth class = argmax
 * of the one-hot label vector.
 */
public final class Evaluator {

    private Evaluator() {}

    /**
     * Runs the model over every (image, label) pair and returns the fraction
     * that were classified correctly.
     *
     * @param model  the network to evaluate (not modified)
     * @param images array of input column-vector Matrices
     * @param labels array of one-hot column-vector Matrices, same length as images
     * @return accuracy in [0.0, 1.0]; returns 0.0 if the arrays are empty
     * @throws IllegalArgumentException if images and labels have different lengths
     */
    public static double evaluate(NeuralNetwork model, Matrix[] images, Matrix[] labels) {
        if (images.length != labels.length) {
            throw new IllegalArgumentException(
                    "images and labels arrays must have the same length, but got " +
                    images.length + " vs " + labels.length);
        }
        if (images.length == 0) return 0.0;

        int correct = 0;
        for (int i = 0; i < images.length; i++) {
            Matrix output = model.predict(images[i]);
            if (argmax(output) == argmax(labels[i])) {
                correct++;
            }
        }
        return (double) correct / images.length;
    }

    /**
     * Returns the index of the largest value in a column-vector Matrix.
     */
    private static int argmax(Matrix m) {
        int rows = m.getRows();
        int best = 0;
        double bestVal = m.get(0, 0);
        for (int i = 1; i < rows; i++) {
            double v = m.get(i, 0);
            if (v > bestVal) {
                bestVal = v;
                best = i;
            }
        }
        return best;
    }
}
