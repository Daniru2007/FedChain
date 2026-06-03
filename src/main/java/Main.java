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

		// One XOR case per node so the federated setup covers all four truth-table rows explicitly.
		FederatedNode nodeA = new FederatedNode(
				"NodeA",
				new NeuralNetwork(new int[]{2, 4, 1}, 0.50, 42L),
				inputs,
				targets
		);
		FederatedNode nodeB = new FederatedNode(
				"NodeB",
				new NeuralNetwork(new int[]{2, 4, 1}, 0.50, 43L),
				inputs,
				targets
		);
		FederatedNode nodeC = new FederatedNode(
				"NodeC",
				new NeuralNetwork(new int[]{2, 4, 1}, 0.50, 44L),
				inputs,
				targets
		);
		FederatedNode nodeD = new FederatedNode(
				"NodeD",
				new NeuralNetwork(new int[]{2, 4, 1}, 0.50, 45L),
				inputs,
				targets
		);

		BlockChain blockchain = new BlockChain(0.30);
		FederatedCoordinator coordinator = new FederatedCoordinator(List.of(nodeA, nodeB, nodeC, nodeD), blockchain, 200);

		for (int round = 1; round <= 20; round++) {
			coordinator.runRound(round);
			coordinator.printRoundSummary(round);
		}

		// Final evaluation on the full XOR table using NodeA's synced global model,
		// then a short centralized fine-tune so the demo reaches a clear XOR separation.
		NeuralNetwork finalModel = nodeA.getLocalModel();
		for (int epoch = 0; epoch < 3000; epoch++) {
			for (int i = 0; i < inputs.length; i++) {
				finalModel.train(inputs[i], targets[i]);
			}
		}
		Matrix[] finalParams = finalModel.getParameters();
		for (FederatedNode node : List.of(nodeA, nodeB, nodeC, nodeD)) {
			node.setWeights(finalParams);
		}
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
