import blockchain.BlockChain;
import math.Matrix;
import ml.FederatedCoordinator;
import ml.FederatedNode;
import ml.NeuralNetwork;

import java.util.List;

public class Main {
    public static void main(String[] args) {
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

        Matrix[] inputsA = {inputs[0], inputs[1]};
        Matrix[] targetsA = {targets[0], targets[1]};

        Matrix[] inputsB = {inputs[2]};
        Matrix[] targetsB = {targets[2]};

        Matrix[] inputsC = {inputs[3]};
        Matrix[] targetsC = {targets[3]};

        Matrix[] inputsD = inputs;
        Matrix[] targetsD = targets;

        FederatedNode nodeA = new FederatedNode(
                "NodeA",
                new NeuralNetwork(new int[]{2, 4, 1}, 0.3, 42L),
                inputsA, targetsA
        );
        FederatedNode nodeB = new FederatedNode(
                "NodeB",
                new NeuralNetwork(new int[]{2, 4, 1}, 0.3, 43L),
                inputsB, targetsB
        );
        FederatedNode nodeC = new FederatedNode(
                "NodeC",
                new NeuralNetwork(new int[]{2, 4, 1}, 0.3, 44L),
                inputsC, targetsC
        );
        FederatedNode nodeD = new FederatedNode(
                "NodeD",
                new NeuralNetwork(new int[]{2, 4, 1}, 0.3, 45L),
                inputsD, targetsD
        );

        BlockChain blockchain = new BlockChain(0.30);
        FederatedCoordinator coordinator = new FederatedCoordinator(List.of(nodeA, nodeB, nodeC, nodeD), blockchain, 1000);

        for (int round = 1; round <= 20; round++) {
            coordinator.runRound(round);
            coordinator.printRoundSummary(round);
        }

        NeuralNetwork finalModel = nodeD.getLocalModel();
        System.out.println("\nFinal Predictions:");
        for (int i = 0; i < inputs.length; i++) {
            double prediction = finalModel.predict(inputs[i]).get(0, 0);
            System.out.printf("[%.0f,%.0f] → %.2f %s%n",
                    inputs[i].get(0, 0),
                    inputs[i].get(1, 0),
                    prediction,
                    ((prediction < 0.2 && targets[i].get(0, 0) == 0.0) || (prediction > 0.8 && targets[i].get(0, 0) == 1.0)) ? "✅" : "");
        }

        System.out.println("\nBlockchain valid: " + blockchain.isChainValid());
        System.out.println("Total blocks: " + blockchain.getChain().size());
    }
}
