package ml;

import blockchain.Block;
import blockchain.BlockChain;
import math.Matrix;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Orchestrates one federated training round.
 *
 * <p>V3 change: each node's update must now pass decentralised consensus via the
 * {@link ConsensusEngine} before it is accepted into FedAvg. The blockchain still
 * records every accepted block; updates that fail consensus are discarded and
 * never reach the chain.
 */
public class FederatedCoordinator {
    private final List<FederatedNode> nodes;
    private final BlockChain blockchain;
    private final ConsensusEngine consensusEngine;
    private final int epochsPerRound;

    /** Snapshot of the current global model; updated after each successful round. */
    private NeuralNetwork globalModel;

    private final List<String> lastAcceptedNodes = new ArrayList<>();
    private final List<String> lastRejectedNodes  = new ArrayList<>();
    private final Map<String, Double> lastLossByNode = new LinkedHashMap<>();

    /**
     * @param nodes           the training nodes participating in federated learning
     * @param blockchain      the shared immutable ledger
     * @param consensusEngine the decentralised validator network
     * @param globalModel     the initial global model (used as baseline by validators in round 1)
     * @param epochsPerRound  local epochs each node trains per round
     */
    public FederatedCoordinator(List<FederatedNode> nodes,
                                BlockChain blockchain,
                                ConsensusEngine consensusEngine,
                                NeuralNetwork globalModel,
                                int epochsPerRound) {
        this.nodes           = List.copyOf(Objects.requireNonNull(nodes, "nodes cannot be null"));
        this.blockchain      = Objects.requireNonNull(blockchain, "blockchain cannot be null");
        this.consensusEngine = Objects.requireNonNull(consensusEngine, "consensusEngine cannot be null");
        this.globalModel     = Objects.requireNonNull(globalModel, "globalModel cannot be null").copy();
        if (epochsPerRound < 0) {
            throw new IllegalArgumentException("epochsPerRound cannot be negative.");
        }
        this.epochsPerRound = epochsPerRound;
    }

    /**
     * Runs one federated round:
     * <ol>
     *   <li>Each node trains locally for {@code epochsPerRound} epochs.</li>
     *   <li>The node's updated weights are submitted to the {@link ConsensusEngine}.</li>
     *   <li>If consensus passes, a block is appended to the blockchain and the
     *       model is added to the FedAvg pool.</li>
     *   <li>FedAvg is applied to all accepted models and the result becomes the
     *       new global model, which is distributed back to every node.</li>
     * </ol>
     */
    public void runRound(int round) {
        lastAcceptedNodes.clear();
        lastRejectedNodes.clear();
        lastLossByNode.clear();

        List<NeuralNetwork> acceptedModels = new ArrayList<>();

        for (FederatedNode node : nodes) {
            node.train(epochsPerRound);

            Matrix[] candidateWeights = node.getWeights();
            double   loss             = node.getLoss();
            lastLossByNode.put(node.getNodeId(), loss);

            // ── Decentralised consensus check ──────────────────────────────
            boolean passed = consensusEngine.reachConsensus(node.getNodeId(), candidateWeights, globalModel);

            if (passed) {
                // Record on the blockchain only after consensus is reached
                Block block = node.createBlock(round, blockchain.getLatestBlock().getHash());
                blockchain.addBlock(block);

                lastAcceptedNodes.add(node.getNodeId());
                acceptedModels.add(node.getLocalModel());
            } else {
                lastRejectedNodes.add(node.getNodeId());
            }
        }

        // ── FedAvg over accepted models only ──────────────────────────────
        if (!acceptedModels.isEmpty()) {
            Matrix[] averaged = FedAvg.aggregate(acceptedModels);
            // Update the global model snapshot
            globalModel.setParameters(averaged);
            // Distribute to all training nodes
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

    /** Returns a copy of the current global model. */
    public NeuralNetwork getGlobalModel() {
        return globalModel.copy();
    }

    public double getGlobalLoss() {
        if (nodes.isEmpty()) return Double.NaN;
        double sum = 0.0;
        for (FederatedNode node : nodes) {
            sum += node.getLoss();
        }
        return sum / nodes.size();
    }
}
