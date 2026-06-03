package ml;

import blockchain.Block;
import blockchain.HashUtil;
import math.Matrix;

import java.util.Objects;

public class FederatedNode {
    private final String nodeId;
    private final NeuralNetwork localModel;
    private final Matrix[] localInputs;
    private final Matrix[] localTargets;
    private double lastLoss = Double.NaN;

    public FederatedNode(String nodeId, NeuralNetwork localModel, Matrix[] localInputs, Matrix[] localTargets) {
        this.nodeId = Objects.requireNonNull(nodeId, "nodeId cannot be null");
        this.localModel = Objects.requireNonNull(localModel, "localModel cannot be null").copy();
        this.localInputs = copyArray(Objects.requireNonNull(localInputs, "localInputs cannot be null"));
        this.localTargets = copyArray(Objects.requireNonNull(localTargets, "localTargets cannot be null"));
        if (this.localInputs.length != this.localTargets.length) {
            throw new IllegalArgumentException("localInputs and localTargets must have the same length.");
        }
    }

    public void train(int epochs) {
        if (epochs < 0) {
            throw new IllegalArgumentException("epochs cannot be negative.");
        }
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < localInputs.length; i++) {
                localModel.train(localInputs[i], localTargets[i]);
            }
            lastLoss = recalculateLoss();
        }
    }

    public Matrix[] getWeights() {
        return localModel.getParameters();
    }

    public String getNodeId() {
        return nodeId;
    }

    public double getLoss() {
        return Double.isNaN(lastLoss) ? recalculateLoss() : lastLoss;
    }

    public void setWeights(Matrix[] weights) {
        localModel.setParameters(weights);
        lastLoss = recalculateLoss();
    }

    public NeuralNetwork getLocalModel() {
        return localModel.copy();
    }

    public Block createBlock(int round, String previousHash) {
        String modelHash = hashParameters(localModel.getParameters());
        return new Block(nodeId, round, modelHash, getLoss(), System.currentTimeMillis(), previousHash);
    }

    private double recalculateLoss() {
        if (localInputs.length == 0) {
            lastLoss = Double.NaN;
            return lastLoss;
        }
        double sum = 0.0;
        for (int i = 0; i < localInputs.length; i++) {
            Matrix out = localModel.predict(localInputs[i]);
            sum += mse(out, localTargets[i]);
        }
        lastLoss = sum / localInputs.length;
        return lastLoss;
    }

    private static double mse(Matrix a, Matrix b) {
        double[][] da = a.getData();
        double[][] db = b.getData();
        double sum = 0.0;
        for (int i = 0; i < da.length; i++) {
            for (int j = 0; j < da[i].length; j++) {
                double d = da[i][j] - db[i][j];
                sum += d * d;
            }
        }
        return sum / Math.max(1, da.length * (da.length == 0 ? 0 : da[0].length));
    }

    private static Matrix[] copyArray(Matrix[] source) {
        Matrix[] copy = new Matrix[source.length];
        System.arraycopy(source, 0, copy, 0, source.length);
        return copy;
    }

    private static String hashParameters(Matrix[] parameters) {
        StringBuilder sb = new StringBuilder();
        for (Matrix matrix : parameters) {
            sb.append(matrix.getRows()).append('x').append(matrix.getCols()).append('|');
            double[][] data = matrix.getData();
            for (double[] row : data) {
                for (double value : row) {
                    sb.append(value).append(',');
                }
            }
            sb.append(';');
        }
        return HashUtil.sha256(sb.toString());
    }
}

