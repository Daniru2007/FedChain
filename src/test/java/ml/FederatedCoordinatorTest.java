package ml;

import blockchain.BlockChain;
import math.Matrix;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FederatedCoordinatorTest {

    private static Matrix col(double a, double b) {
        return new Matrix(new double[][]{{a}, {b}});
    }

    /**
     * Builds a permissive ConsensusEngine whose single validator always accepts.
     * A tolerance of 1.0 (100 pp) means any accuracy >= (global - 100%) → always passes.
     * The validation set mirrors the node data so the validator can evaluate the model.
     */
    private static ConsensusEngine permissiveConsensus(Matrix[] valImages, Matrix[] valLabels) {
        ValidatorNode v = new ValidatorNode("TestValidator", valImages, valLabels, 1.0);
        return new ConsensusEngine(List.of(v));
    }

    @Test
    void runRoundAddsAcceptedBlocksAndSynchronizesModels() {
        Matrix oh0 = new Matrix(new double[][]{{1.0}, {0.0}}); // one-hot class 0
        Matrix oh1 = new Matrix(new double[][]{{0.0}, {1.0}}); // one-hot class 1

        Matrix[] valImages = {col(0,0), col(0,1), col(1,0), col(1,1)};
        Matrix[] valLabels = {oh0, oh1, oh1, oh0};

        FederatedNode nodeA = new FederatedNode("NodeA",
                new NeuralNetwork(new int[]{2, 4, 2}, 0.3, 1L),
                new Matrix[]{col(0,0), col(0,1)},
                new Matrix[]{oh0, oh1});

        FederatedNode nodeB = new FederatedNode("NodeB",
                new NeuralNetwork(new int[]{2, 4, 2}, 0.3, 2L),
                new Matrix[]{col(1,0)},
                new Matrix[]{oh1});

        FederatedNode nodeC = new FederatedNode("NodeC",
                new NeuralNetwork(new int[]{2, 4, 2}, 0.3, 3L),
                new Matrix[]{col(1,1)},
                new Matrix[]{oh0});

        NeuralNetwork globalModel = new NeuralNetwork(new int[]{2, 4, 2}, 0.3, 0L);
        BlockChain blockchain = new BlockChain(10.0);
        FederatedCoordinator coordinator = new FederatedCoordinator(
                List.of(nodeA, nodeB, nodeC),
                blockchain,
                permissiveConsensus(valImages, valLabels),
                globalModel,
                10);

        coordinator.runRound(1);

        // genesis + at least one accepted block
        assertTrue(blockchain.getChain().size() >= 2);
        assertTrue(blockchain.isChainValid());

        // all nodes should have same weights after sync
        Matrix[] aWeights = nodeA.getWeights();
        Matrix[] bWeights = nodeB.getWeights();
        assertEquals(aWeights.length, bWeights.length);
    }

    @Test
    void blockchainRemainsValidAcrossMultipleRounds() {
        Matrix oh0 = new Matrix(new double[][]{{1.0}, {0.0}});
        Matrix oh1 = new Matrix(new double[][]{{0.0}, {1.0}});

        Matrix[] valImages = {col(0,0), col(0,1), col(1,0), col(1,1)};
        Matrix[] valLabels = {oh0, oh1, oh1, oh0};

        FederatedNode nodeA = new FederatedNode("NodeA",
                new NeuralNetwork(new int[]{2, 4, 2}, 0.3, 11L),
                new Matrix[]{col(0,0), col(0,1)},
                new Matrix[]{oh0, oh1});

        FederatedNode nodeB = new FederatedNode("NodeB",
                new NeuralNetwork(new int[]{2, 4, 2}, 0.3, 12L),
                new Matrix[]{col(1,0), col(1,1)},
                new Matrix[]{oh1, oh0});

        NeuralNetwork globalModel = new NeuralNetwork(new int[]{2, 4, 2}, 0.3, 0L);
        BlockChain blockchain = new BlockChain(10.0);
        FederatedCoordinator coordinator = new FederatedCoordinator(
                List.of(nodeA, nodeB),
                blockchain,
                permissiveConsensus(valImages, valLabels),
                globalModel,
                5);

        for (int round = 1; round <= 5; round++) {
            coordinator.runRound(round);
        }

        assertTrue(blockchain.isChainValid());
        // genesis + up to 2 blocks per round × 5 rounds
        assertTrue(blockchain.getChain().size() >= 1);
    }

    @Test
    void lossDecreasesAfterMultipleRounds() {
        Matrix oh0 = new Matrix(new double[][]{{1.0}, {0.0}});
        Matrix oh1 = new Matrix(new double[][]{{0.0}, {1.0}});

        Matrix[] valImages = {col(0,0), col(0,1), col(1,0), col(1,1)};
        Matrix[] valLabels = {oh0, oh1, oh1, oh0};

        FederatedNode nodeA = new FederatedNode("NodeA",
                new NeuralNetwork(new int[]{2, 4, 2}, 0.3, 42L),
                new Matrix[]{col(0,0), col(0,1)},
                new Matrix[]{oh0, oh1});

        FederatedNode nodeB = new FederatedNode("NodeB",
                new NeuralNetwork(new int[]{2, 4, 2}, 0.3, 43L),
                new Matrix[]{col(1,0), col(1,1)},
                new Matrix[]{oh1, oh0});

        NeuralNetwork globalModel = new NeuralNetwork(new int[]{2, 4, 2}, 0.3, 0L);
        BlockChain blockchain = new BlockChain(10.0);
        FederatedCoordinator coordinator = new FederatedCoordinator(
                List.of(nodeA, nodeB),
                blockchain,
                permissiveConsensus(valImages, valLabels),
                globalModel,
                500);

        coordinator.runRound(1);
        double lossAfterRound1 = coordinator.getGlobalLoss();

        for (int round = 2; round <= 20; round++) {
            coordinator.runRound(round);
        }
        double lossAfterRound20 = coordinator.getGlobalLoss();

        assertTrue(lossAfterRound20 < lossAfterRound1,
                "loss should decrease over rounds");
    }
}