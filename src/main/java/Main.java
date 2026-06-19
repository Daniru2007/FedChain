import blockchain.BlockChain;
import math.Matrix;
import ml.*;

import java.util.Arrays;
import java.util.List;

public class Main {

    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------
    private static final String MNIST_DIR    = "data/mnist";
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

    /** Number of independent validator nodes. */
    private static final int    NUM_VALIDATORS    = 3;

    /**
     * How far below the global model accuracy (as a fraction) a candidate is still
     * allowed to be. e.g. 0.05 means the candidate must be within 5 pp of the global.
     */
    private static final double TOLERANCE_MARGIN = 0.05;

    /**
     * How many of the 10,000 test images to set aside as the shared validation set
     * (used by validators). The remaining images form the held-out test set.
     */
    private static final int VALIDATION_SIZE = 2000;

    // -----------------------------------------------------------------------

    public static void main(String[] args) throws Exception {
        System.out.println("=== FedChain V3 — Decentralised Consensus with MNIST ===\n");

        // ── Load data ──────────────────────────────────────────────────────
        System.out.println("Loading MNIST training data…");
        Matrix[] trainImages = MNISTLoader.loadImages(TRAIN_IMAGES);
        Matrix[] trainLabels = MNISTLoader.loadLabels(TRAIN_LABELS);
        System.out.printf("  Training samples : %,d%n", trainImages.length);

        System.out.println("Loading MNIST test/validation data…");
        Matrix[] allTestImages = MNISTLoader.loadImages(TEST_IMAGES);
        Matrix[] allTestLabels = MNISTLoader.loadLabels(TEST_LABELS);
        System.out.printf("  Total test samples : %,d%n", allTestImages.length);

        // ── Split into validation set (for validators) and test set ────────
        Matrix[] validImages = Arrays.copyOfRange(allTestImages, 0, VALIDATION_SIZE);
        Matrix[] validLabels = Arrays.copyOfRange(allTestLabels, 0, VALIDATION_SIZE);
        Matrix[] testImages  = Arrays.copyOfRange(allTestImages, VALIDATION_SIZE, allTestImages.length);
        Matrix[] testLabels  = Arrays.copyOfRange(allTestLabels, VALIDATION_SIZE, allTestLabels.length);
        System.out.printf("  Validation set   : %,d samples%n", validImages.length);
        System.out.printf("  Test set         : %,d samples%n%n", testImages.length);

        // ── Partition training data across nodes ───────────────────────────
        int total     = trainImages.length;
        int chunkSize = total / NUM_NODES;

        // Shared initial global model — all nodes start from the same weights
        NeuralNetwork globalModel = new NeuralNetwork(ARCHITECTURE, LEARNING_RATE, 0L);

        FederatedNode[] nodes = new FederatedNode[NUM_NODES];
        for (int n = 0; n < NUM_NODES; n++) {
            int from = n * chunkSize;
            int to   = (n == NUM_NODES - 1) ? total : from + chunkSize;

            Matrix[] nodeImages = Arrays.copyOfRange(trainImages, from, to);
            Matrix[] nodeLabels = Arrays.copyOfRange(trainLabels, from, to);

            // Each node gets a copy of the shared initial global model
            nodes[n] = new FederatedNode(
                    "Node" + (char) ('A' + n),
                    globalModel.copy(),
                    nodeImages,
                    nodeLabels
            );
            System.out.printf("Node%c → samples [%,d – %,d) (%,d samples)%n",
                    (char) ('A' + n), from, to, to - from);
        }
        System.out.println();

        // ── Validators ─────────────────────────────────────────────────────
        // All validators share the same validation dataset
        ValidatorNode[] validators = new ValidatorNode[NUM_VALIDATORS];
        for (int v = 0; v < NUM_VALIDATORS; v++) {
            validators[v] = new ValidatorNode(
                    "Validator" + (char) ('1' + v),
                    validImages,
                    validLabels,
                    TOLERANCE_MARGIN
            );
        }
        System.out.printf("%d validator node(s) online (validation set: %,d samples, tolerance: %.0f%%)%n%n",
                NUM_VALIDATORS, validImages.length, TOLERANCE_MARGIN * 100);

        // ── Consensus engine + blockchain + coordinator ────────────────────
        ConsensusEngine  consensusEngine = new ConsensusEngine(List.of(validators));
        BlockChain       blockchain      = new BlockChain(/* lossThreshold */ 5.0);
        FederatedCoordinator coordinator = new FederatedCoordinator(
                List.of(nodes), blockchain, consensusEngine, globalModel, EPOCHS_PER_ROUND);

        // ── Federated training ─────────────────────────────────────────────
        for (int round = 1; round <= ROUNDS; round++) {
            System.out.printf("--- Round %d/%d ---%n", round, ROUNDS);
            coordinator.runRound(round);
            coordinator.printRoundSummary(round);

            double accuracy = Evaluator.evaluate(coordinator.getGlobalModel(), testImages, testLabels);
            System.out.printf("  ▶ Test accuracy after round %d: %.2f%%%n%n",
                    round, accuracy * 100.0);
        }

        // ── Final summary ──────────────────────────────────────────────────
        double finalAccuracy = Evaluator.evaluate(coordinator.getGlobalModel(), testImages, testLabels);
        System.out.printf("=== Final Test Accuracy : %.2f%% ===%n", finalAccuracy * 100.0);
        System.out.println("Blockchain valid        : " + blockchain.isChainValid());
        System.out.println("Total blocks            : " + blockchain.getChain().size());
    }
}
