package ml;

import math.Matrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertFalse;
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

    @Test
    void saveAndLoadPreservePredictions() throws Exception {
        NeuralNetwork nn = new NeuralNetwork(new int[]{2, 3, 1}, 0.1);
        Matrix input = new Matrix(new double[][]{{0.2}, {0.8}});
        Matrix p1 = nn.predict(input);

        java.nio.file.Path tmp = java.nio.file.Files.createTempFile("nn", ".bin");
        String path = tmp.toAbsolutePath().toString();
        nn.save(path);

        NeuralNetwork nn2 = NeuralNetwork.load(path);
        Matrix p2 = nn2.predict(input);
        assertEquals(p1.get(0, 0), p2.get(0, 0), 1e-12);
    }

    @Test
    void saveBareNameGoesIntoModelsDirectory() throws Exception {
        NeuralNetwork nn = new NeuralNetwork(new int[]{1, 1}, 0.1);
        java.nio.file.Path modelsDir = java.nio.file.Paths.get(System.getProperty("user.dir"), "models");
        java.nio.file.Files.createDirectories(modelsDir);
        java.nio.file.Path file = modelsDir.resolve("test-model.bin");
        java.nio.file.Files.deleteIfExists(file);

        nn.save("test-model.bin");

        assertTrue(java.nio.file.Files.exists(file));
        assertTrue(java.nio.file.Files.size(file) > 0);
    }

    @Test
    void loadFromExplicitPathStillWorks() throws Exception {
        NeuralNetwork nn = new NeuralNetwork(new int[]{1, 1}, 0.1);
        java.nio.file.Path tmp = java.nio.file.Files.createTempFile("nn-explicit", ".bin");
        nn.save(tmp.toString());

        NeuralNetwork loaded = NeuralNetwork.load(tmp.toString());
        assertNotNull(loaded);

        Matrix input = new Matrix(new double[][]{{0.4}});
        assertEquals(nn.predict(input).get(0, 0), loaded.predict(input).get(0, 0), 1e-12);
    }

    @Test
    void xorDemoLearnsTheXorPattern() {
        NeuralNetwork nn = new NeuralNetwork(new int[]{2, 4, 1}, 0.5, 42L);
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

        double initialLoss = 0.0;
        for (int i = 0; i < inputs.length; i++) {
            initialLoss += mse(nn.predict(inputs[i]), targets[i]);
        }
        initialLoss /= inputs.length;

        for (int epoch = 0; epoch < 10_000; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                nn.train(inputs[i], targets[i]);
            }
        }

        double finalLoss = 0.0;
        double[] predictions = new double[4];
        for (int i = 0; i < inputs.length; i++) {
            Matrix out = nn.predict(inputs[i]);
            predictions[i] = out.get(0, 0);
            finalLoss += mse(out, targets[i]);
        }
        finalLoss /= inputs.length;

        assertTrue(finalLoss < initialLoss, "XOR training should reduce loss");
        assertTrue(finalLoss < 0.15, "XOR demo should get reasonably low error");
        assertTrue(predictions[0] < 0.4);
        assertTrue(predictions[1] > 0.6);
        assertTrue(predictions[2] > 0.6);
        assertTrue(predictions[3] < 0.4);
    }
}

