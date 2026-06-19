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
        
        int totalImages = localInputs.length;
        int printInterval = Math.max(1, totalImages / 10); // Update the log 10 times per epoch
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < totalImages; i++) {
                localModel.train(localInputs[i], localTargets[i]);
                
                // Print progress periodically
                if ((i + 1) % printInterval == 0 || i == totalImages - 1) {
                    double progress = ((double) (i + 1) / totalImages) * 100;
                    System.out.printf("[%s] Epoch %d/%d - Progress: %5.1f%% - Live Loss: %.4f\n", 
                                      nodeId, epoch + 1, epochs, progress, localModel.getLoss());
                }
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
            sum += localModel.getLoss(localInputs[i], localTargets[i]);
        }
        lastLoss = sum / localInputs.length;
        return lastLoss;
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

