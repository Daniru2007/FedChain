import blockchain.BlockChain;
import math.Matrix;
import ml.MNISTLoader;
import ml.NeuralNetwork;
import ml.Evaluator;
import network.NodeLauncher;
import network.Peer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Main {

    private static final String MNIST_DIR = "data/mnist";
    private static final String TRAIN_IMAGES = MNIST_DIR + "/train-images-idx3-ubyte.gz";
    private static final String TRAIN_LABELS = MNIST_DIR + "/train-labels-idx1-ubyte.gz";
    private static final String TEST_IMAGES = MNIST_DIR + "/t10k-images-idx3-ubyte.gz";
    private static final String TEST_LABELS = MNIST_DIR + "/t10k-labels-idx1-ubyte.gz";

    private static final int[] ARCHITECTURE = {784, 128, 64, 10};
    private static final double LEARNING_RATE = 0.01;

    public static void main(String[] args) throws Exception {
        System.out.println("=== FedChain V4 — Decentralized Async P2P Network ===");

        // 1. Load Data
        Matrix[] trainImages = MNISTLoader.loadImages(TRAIN_IMAGES);
        Matrix[] trainLabels = MNISTLoader.loadLabels(TRAIN_LABELS);
        Matrix[] testImages = MNISTLoader.loadImages(TEST_IMAGES);
        Matrix[] testLabels = MNISTLoader.loadLabels(TEST_LABELS);

        // 2. Define the exact network topology
        List<Peer> fullyConnectedNetwork = new ArrayList<>();
        fullyConnectedNetwork.add(new Peer("NodeA", "localhost", 8081));
        fullyConnectedNetwork.add(new Peer("NodeB", "localhost", 8082));
        fullyConnectedNetwork.add(new Peer("NodeC", "localhost", 8083));
        fullyConnectedNetwork.add(new Peer("NodeD", "localhost", 8084));
        fullyConnectedNetwork.add(new Peer("Validator1", "localhost", 8091));
        fullyConnectedNetwork.add(new Peer("Validator2", "localhost", 8092));
        fullyConnectedNetwork.add(new Peer("Validator3", "localhost", 8093));

        // Shared initial state so they all start on the same page
        NeuralNetwork globalModel = new NeuralNetwork(ARCHITECTURE, LEARNING_RATE, 0L);
        BlockChain genesisChain = new BlockChain(5.0);

        List<NodeLauncher> allNodes = new ArrayList<>();

        // 3. Launch Trainers
        for (int i = 0; i < 4; i++) {
            String nodeId = "Node" + (char) ('A' + i);
            int port = 8081 + i;
            int from = i * 15000;
            int to = from + 15000;
            
            // Just in case MNIST is slightly smaller
            to = Math.min(to, trainImages.length); 

            Matrix[] nodeImages = Arrays.copyOfRange(trainImages, from, to);
            Matrix[] nodeLabels = Arrays.copyOfRange(trainLabels, from, to);

            NodeLauncher trainer = new NodeLauncher(
                nodeId, port, NodeLauncher.Role.TRAINER, fullyConnectedNetwork,
                globalModel.copy(), new BlockChain(5.0), nodeImages, nodeLabels,
                3, 2 // Threshold: 3 updates, Majority: 2 validators
            );
            allNodes.add(trainer);
        }

        // 4. Launch Validators
        for (int i = 0; i < 3; i++) {
            String valId = "Validator" + (i + 1);
            int port = 8091 + i;

            // Validators get the full test set
            NodeLauncher validator = new NodeLauncher(
                valId, port, NodeLauncher.Role.VALIDATOR, fullyConnectedNetwork,
                globalModel.copy(), new BlockChain(5.0), testImages, testLabels,
                3, 2
            );
            allNodes.add(validator);
        }

        // 5. Start them all!
        System.out.println("Starting network... (Press Ctrl+C to stop)");
        for (NodeLauncher node : allNodes) {
            node.start();
        }

        // 6. Monitor accuracy from NodeA's perspective continuously
        NodeLauncher monitorNode = allNodes.get(0);
        while (true) {
            Thread.sleep(10000); // Check every 10 seconds
            
            NeuralNetwork currentGlobal = monitorNode.getTrainingNode().getCoreNode().getLocalModel();
            double acc = Evaluator.evaluate(currentGlobal, testImages, testLabels);
            System.out.println("\n[SYSTEM MONITOR] Current Test Accuracy (from NodeA's view): " + (acc * 100.0) + "%\n");
        }
    }
}
