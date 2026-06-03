package ml;

import math.Matrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NeuralNetworkTest {

    private static double mse(Matrix a, Matrix b) {
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

    @Test
    void forwardDoesNotMutateInput() {
        NeuralNetwork nn = new NeuralNetwork(new int[]{2, 2, 1}, 0.1);
        Matrix input = new Matrix(new double[][]{{0.5}, {0.25}});
        double before = input.get(0, 0);
        double before2 = input.get(1, 0);

        nn.forward(input);

        assertEquals(before, input.get(0, 0));
        assertEquals(before2, input.get(1, 0));
    }

    @Test
    void predictIsDeterministicWhenNotTraining() {
        NeuralNetwork nn = new NeuralNetwork(new int[]{2, 3, 1}, 0.1);
        Matrix input = new Matrix(new double[][]{{0.1}, {0.9}});
        Matrix p1 = nn.predict(input);
        Matrix p2 = nn.predict(input);
        assertEquals(p1.get(0, 0), p2.get(0, 0), 1e-12);
    }

    @Test
    void trainReducesMseForSingleZeroExample() {
        NeuralNetwork nn = new NeuralNetwork(new int[]{1, 1}, 1.0);
        Matrix input = new Matrix(new double[][]{{0.0}});
        Matrix target = new Matrix(new double[][]{{0.0}});

        double before = mse(nn.predict(input), target);
        for (int i = 0; i < 20; i++) {
            nn.train(input, target);
        }
        double after = mse(nn.predict(input), target);

        assertTrue(after < before, "MSE should decrease after training");
    }
}

