import blockchain.BlockChain;
import math.Matrix;
import ml.*;

import java.util.Arrays;
import java.util.List;

public class Main {

    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------
    /** Path to the directory containing the four raw MNIST IDX files. */
    private static final String MNIST_DIR = "data/mnist";

    private static final String TRAIN_IMAGES = MNIST_DIR + "/train-images-idx3-ubyte.gz";
    private static final String TRAIN_LABELS = MNIST_DIR + "/train-labels-idx1-ubyte.gz";
    private static final String TEST_IMAGES  = MNIST_DIR + "/t10k-images-idx3-ubyte.gz";
    private static final String TEST_LABELS  = MNIST_DIR + "/t10k-labels-idx1-ubyte.gz";

    /** Number of federated rounds. */
    private static final int ROUNDS = 10;

    /** Local epochs each node trains per round. */
    private static final int EPOCHS_PER_ROUND = 1;

    /** Network architecture: 784 inputs → 128 hidden → 64 hidden → 10 outputs. */
    private static final int[] ARCHITECTURE = {784, 128, 64, 10};

    private static final double LEARNING_RATE = 0.01;
    private static final int    NUM_NODES     = 4;

    // -----------------------------------------------------------------------

    public static void main(String[] args) throws Exception {
        System.out.println("=== FedChain V2 — MNIST Federated Learning ===\n");

        // ── Load data ──────────────────────────────────────────────────────
        System.out.println("Loading MNIST training data…");
        Matrix[] trainImages = MNISTLoader.loadImages(TRAIN_IMAGES);
        Matrix[] trainLabels = MNISTLoader.loadLabels(TRAIN_LABELS);
        System.out.printf("  Training samples : %,d%n", trainImages.length);

        System.out.println("Loading MNIST test data…");
        Matrix[] testImages = MNISTLoader.loadImages(TEST_IMAGES);
        Matrix[] testLabels = MNISTLoader.loadLabels(TEST_LABELS);
        System.out.printf("  Test samples     : %,d%n%n", testImages.length);

        // ── Partition training data across nodes ───────────────────────────
        int total     = trainImages.length;
        int chunkSize = total / NUM_NODES;

        FederatedNode[] nodes = new FederatedNode[NUM_NODES];
        for (int n = 0; n < NUM_NODES; n++) {
            int from = n * chunkSize;
            int to   = (n == NUM_NODES - 1) ? total : from + chunkSize;

            Matrix[] nodeImages = Arrays.copyOfRange(trainImages, from, to);
            Matrix[] nodeLabels = Arrays.copyOfRange(trainLabels, from, to);

            nodes[n] = new FederatedNode(
                    "Node" + (char) ('A' + n),
                    new NeuralNetwork(ARCHITECTURE, LEARNING_RATE, 42L + n),
                    nodeImages,
                    nodeLabels
            );
            System.out.printf("Node%c  → samples [%,d – %,d) (%,d samples)%n",
                    (char) ('A' + n), from, to, to - from);
        }
        System.out.println();

        // ── Blockchain + coordinator ───────────────────────────────────────
        BlockChain blockchain   = new BlockChain(/* lossThreshold */ 5.0);
        FederatedCoordinator coordinator =
                new FederatedCoordinator(List.of(nodes), blockchain, EPOCHS_PER_ROUND);

        // ── Federated training ─────────────────────────────────────────────
        // Grab the global model reference from any node for evaluation.
        // After each round, all nodes share the same averaged weights.
        for (int round = 1; round <= ROUNDS; round++) {
            coordinator.runRound(round);
            coordinator.printRoundSummary(round);

            // Evaluate the global model on the test set after each round
            NeuralNetwork globalModel = nodes[0].getLocalModel();
            double accuracy = Evaluator.evaluate(globalModel, testImages, testLabels);
            System.out.printf("  ▶ Test accuracy after round %d: %.2f%%%n%n",
                    round, accuracy * 100.0);
        }

        // ── Final summary ──────────────────────────────────────────────────
        NeuralNetwork finalModel = nodes[0].getLocalModel();
        double finalAccuracy = Evaluator.evaluate(finalModel, testImages, testLabels);
        System.out.printf("=== Final Test Accuracy : %.2f%% ===%n", finalAccuracy * 100.0);
        System.out.println("Blockchain valid        : " + blockchain.isChainValid());
        System.out.println("Total blocks            : " + blockchain.getChain().size());
    }
}
