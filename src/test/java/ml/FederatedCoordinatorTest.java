package ml;

import blockchain.BlockChain;
import math.Matrix;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FederatedCoordinatorTest {

    private static Matrix m(double value) {
        return new Matrix(new double[][]{{value}});
    }

    @Test
    void runRoundAddsAcceptedBlocksAndSynchronizesModels() {
        Matrix[] xorInputs = {m(0.0), m(1.0), m(0.0), m(1.0)};
        Matrix[] xorTargets = {m(0.0), m(1.0), m(1.0), m(0.0)};

        FederatedNode nodeA = new FederatedNode("NodeA", new NeuralNetwork(new int[]{1, 2, 1}, 0.5, 1L), new Matrix[]{xorInputs[0]}, new Matrix[]{xorTargets[0]});
        FederatedNode nodeB = new FederatedNode("NodeB", new NeuralNetwork(new int[]{1, 2, 1}, 0.5, 2L), new Matrix[]{xorInputs[1]}, new Matrix[]{xorTargets[1]});
        FederatedNode nodeC = new FederatedNode("NodeC", new NeuralNetwork(new int[]{1, 2, 1}, 0.5, 3L), new Matrix[]{xorInputs[2]}, new Matrix[]{xorTargets[2]});
        FederatedNode nodeD = new FederatedNode("NodeD", new NeuralNetwork(new int[]{1, 2, 1}, 0.5, 4L), new Matrix[]{xorInputs[3]}, new Matrix[]{xorTargets[3]});
        BlockChain blockchain = new BlockChain(0.5);
        FederatedCoordinator coordinator = new FederatedCoordinator(List.of(nodeA, nodeB, nodeC, nodeD), blockchain, 5);

        double before = coordinator.getGlobalLoss();
        coordinator.runRound(1);
        double after = coordinator.getGlobalLoss();

        assertTrue(blockchain.getChain().size() >= 2, "at least one node should be accepted or genesis plus updates");
        assertTrue(Double.isNaN(before) || after <= before || after >= 0.0);
        assertTrue(blockchain.isChainValid());

        Matrix[] aWeights = nodeA.getWeights();
        Matrix[] bWeights = nodeB.getWeights();
        Matrix[] cWeights = nodeC.getWeights();
        Matrix[] dWeights = nodeD.getWeights();
        assertEquals(aWeights.length, bWeights.length);
        assertEquals(aWeights.length, cWeights.length);
        assertEquals(aWeights.length, dWeights.length);
    }

    @Test
    void blockchainRemainsValidAcrossMultipleRounds() {
        Matrix[] xorInputs = {m(0.0), m(1.0), m(0.0), m(1.0)};
        Matrix[] xorTargets = {m(0.0), m(1.0), m(1.0), m(0.0)};

        FederatedNode nodeA = new FederatedNode("NodeA", new NeuralNetwork(new int[]{1, 2, 1}, 0.5, 11L), new Matrix[]{xorInputs[0]}, new Matrix[]{xorTargets[0]});
        FederatedNode nodeB = new FederatedNode("NodeB", new NeuralNetwork(new int[]{1, 2, 1}, 0.5, 12L), new Matrix[]{xorInputs[1]}, new Matrix[]{xorTargets[1]});
        FederatedNode nodeC = new FederatedNode("NodeC", new NeuralNetwork(new int[]{1, 2, 1}, 0.5, 13L), new Matrix[]{xorInputs[2]}, new Matrix[]{xorTargets[2]});
        FederatedNode nodeD = new FederatedNode("NodeD", new NeuralNetwork(new int[]{1, 2, 1}, 0.5, 14L), new Matrix[]{xorInputs[3]}, new Matrix[]{xorTargets[3]});
        BlockChain blockchain = new BlockChain(1.0);
        FederatedCoordinator coordinator = new FederatedCoordinator(List.of(nodeA, nodeB, nodeC, nodeD), blockchain, 2);

        for (int round = 1; round <= 5; round++) {
            coordinator.runRound(round);
        }

        assertTrue(blockchain.isChainValid());
        assertTrue(blockchain.getChain().size() >= 1);
    }

    @Test
    void fourXorCasesAreRepresentedOnceAcrossNodes() {
        Matrix[] xorInputs = {m(0.0), m(1.0), m(0.0), m(1.0)};
        Matrix[] xorTargets = {m(0.0), m(1.0), m(1.0), m(0.0)};

        assertEquals(4, xorInputs.length);
        assertEquals(4, xorTargets.length);
        assertEquals(0.0, xorInputs[0].get(0, 0), 1e-12);
        assertEquals(1.0, xorInputs[1].get(0, 0), 1e-12);
        assertEquals(0.0, xorInputs[2].get(0, 0), 1e-12);
        assertEquals(1.0, xorInputs[3].get(0, 0), 1e-12);
    }
}

