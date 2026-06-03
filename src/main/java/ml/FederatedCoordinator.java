package ml;

import blockchain.Block;
import blockchain.BlockChain;
import math.Matrix;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

public class FederatedCoordinator {
    private final List<FederatedNode> nodes;
    private final BlockChain blockchain;
    private final int epochsPerRound;

    private final List<String> lastAcceptedNodes = new ArrayList<>();
    private final List<String> lastRejectedNodes = new ArrayList<>();
    private final Map<String, Double> lastLossByNode = new LinkedHashMap<>();

    public FederatedCoordinator(List<FederatedNode> nodes, BlockChain blockchain, int epochsPerRound) {
        this.nodes = List.copyOf(Objects.requireNonNull(nodes, "nodes cannot be null"));
        this.blockchain = Objects.requireNonNull(blockchain, "blockchain cannot be null");
        if (epochsPerRound < 0) {
            throw new IllegalArgumentException("epochsPerRound cannot be negative.");
        }
        this.epochsPerRound = epochsPerRound;
    }

    public void runRound(int round) {
        lastAcceptedNodes.clear();
        lastRejectedNodes.clear();
        lastLossByNode.clear();

        List<NeuralNetwork> acceptedModels = new ArrayList<>();

        for (FederatedNode node : nodes) {
            node.train(epochsPerRound);
            Block block = node.createBlock(round, blockchain.getLatestBlock().getHash());
            boolean accepted = blockchain.addBlock(block);

            double loss = node.getLoss();
            lastLossByNode.put(block.getNodeId(), loss);
            if (accepted) {
                lastAcceptedNodes.add(block.getNodeId());
                acceptedModels.add(node.getLocalModel());
            } else {
                lastRejectedNodes.add(block.getNodeId());
            }
        }

        if (!acceptedModels.isEmpty()) {
            Matrix[] averaged = FedAvg.aggregate(acceptedModels);
            for (FederatedNode node : nodes) {
                node.setWeights(averaged);
            }
        }
    }

    public void printRoundSummary(int round) {
        System.out.printf("Round %d: ", round);
        if (lastAcceptedNodes.isEmpty()) {
            System.out.print("no valid updates");
        } else {
            for (String nodeId : lastAcceptedNodes) {
                System.out.printf("%s accepted(loss=%.4f) ", nodeId, lastLossByNode.get(nodeId));
            }
        }
        for (String nodeId : lastRejectedNodes) {
            System.out.printf("%s rejected(loss=%.4f) ", nodeId, lastLossByNode.get(nodeId));
        }
        System.out.printf("| blockchain length=%d%n", blockchain.getChain().size());
        for (FederatedNode node : nodes) {
            System.out.printf("  %s current loss=%.6f%n", node.getNodeId(), node.getLoss());
        }
    }

    public double getGlobalLoss() {
        if (nodes.isEmpty()) {
            return Double.NaN;
        }
        double sum = 0.0;
        for (FederatedNode node : nodes) {
            sum += node.getLoss();
        }
        return sum / nodes.size();
    }
}

