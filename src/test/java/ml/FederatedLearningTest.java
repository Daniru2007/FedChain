package ml;

import blockchain.Block;
import math.Matrix;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FederatedLearningTest {

    private static Matrix m(double value) {
        return new Matrix(new double[][]{{value}});
    }

    @Test
    void fedAvgAveragesParametersExactly() {
        NeuralNetwork n1 = new NeuralNetwork(new int[]{1, 1}, 0.1, 1L);
        NeuralNetwork n2 = new NeuralNetwork(new int[]{1, 1}, 0.1, 2L);

        n1.setParameters(new Matrix[]{m(1.0), m(2.0)});
        n2.setParameters(new Matrix[]{m(3.0), m(4.0)});

        Matrix[] avg = FedAvg.aggregate(List.of(n1, n2));
        assertEquals(2, avg.length);
        assertEquals(2.0, avg[0].get(0, 0), 1e-12);
        assertEquals(3.0, avg[1].get(0, 0), 1e-12);
    }

    @Test
    void federatedNodeTrainsAndReportsLoss() {
        NeuralNetwork model = new NeuralNetwork(new int[]{1, 1}, 0.5, 42L);
        FederatedNode node = new FederatedNode(
                "NodeA",
                model,
                new Matrix[]{m(0.0)},
                new Matrix[]{m(0.0)}
        );

        double before = node.getLoss();
        node.train(25);
        double after = node.getLoss();

        assertTrue(after < before, "local training should reduce loss");
    }

    @Test
    void federatedNodeCreatesBlockFromCurrentModel() {
        NeuralNetwork model = new NeuralNetwork(new int[]{1, 1}, 0.5, 7L);
        FederatedNode node = new FederatedNode(
                "NodeB",
                model,
                new Matrix[]{m(1.0)},
                new Matrix[]{m(1.0)}
        );
        node.train(1);

        Block block = node.createBlock(3, "prev-hash");
        assertEquals("NodeB", block.getNodeId());
        assertEquals(3, block.getRound());
        assertEquals("prev-hash", block.getPreviousHash());
        assertTrue(block.getHash().length() == 64);
    }

    @Test
    void nodeCanAdoptAggregatedParameters() {
        NeuralNetwork model = new NeuralNetwork(new int[]{1, 1}, 0.1, 10L);
        FederatedNode node = new FederatedNode(
                "NodeC",
                model,
                new Matrix[]{m(1.0)},
                new Matrix[]{m(0.0)}
        );

        Matrix[] params = new Matrix[]{m(5.0), m(-1.0)};
        node.setWeights(params);

        Matrix[] current = node.getWeights();
        assertEquals(5.0, current[0].get(0, 0), 1e-12);
        assertEquals(-1.0, current[1].get(0, 0), 1e-12);
    }

    @Test
    void federatedNodeExposesStableIdAndLoss() {
        NeuralNetwork model = new NeuralNetwork(new int[]{1, 1}, 0.5, 3L);
        FederatedNode node = new FederatedNode(
                "NodeX",
                model,
                new Matrix[]{m(0.0)},
                new Matrix[]{m(0.0)}
        );

        assertEquals("NodeX", node.getNodeId());
        double lossBefore = node.getLoss();
        node.train(1);
        double lossAfter = node.getLoss();
        assertTrue(lossAfter >= 0.0);
        assertTrue(!Double.isNaN(lossBefore));
    }

    @Test
    void setWeightsRejectsWrongShape() {
        NeuralNetwork model = new NeuralNetwork(new int[]{1, 1}, 0.1, 4L);
        FederatedNode node = new FederatedNode(
                "NodeY",
                model,
                new Matrix[]{m(0.0)},
                new Matrix[]{m(0.0)}
        );

        try {
            node.setWeights(new Matrix[]{new Matrix(2, 1)});
        } catch (IllegalArgumentException expected) {
            return;
        }
        throw new AssertionError("Expected IllegalArgumentException for wrong parameter count/shape");
    }
}

