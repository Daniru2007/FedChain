package network;

import blockchain.Block;
import blockchain.BlockChain;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import math.Matrix;
import ml.FedAvg;
import ml.NeuralNetwork;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * Handles the leader election process when the pending pool hits the threshold.
 * Avoids race conditions by requiring the pool hashes to match.
 */
public class LeaderElection {
    
    private final String localNodeId;
    private final GossipNode gossipNode;
    private final PendingPool pendingPool;
    private final BlockChain blockchain;
    private final AsyncTrainingNode trainingNode; // to notify of new model
    private final AsyncValidatorNode validatorNode; // to notify of new model
    private final Gson gson;
    
    // tracks which nodes claim to be ready with which pool hash
    // poolHash -> list of ready nodeIds
    private final Map<String, List<String>> readyNodes = new ConcurrentHashMap<>();
    
    private volatile boolean electionInProgress = false;

    public LeaderElection(String localNodeId, GossipNode gossipNode, PendingPool pendingPool, 
                          BlockChain blockchain, AsyncTrainingNode trainingNode, AsyncValidatorNode validatorNode) {
        this.localNodeId = localNodeId;
        this.gossipNode = gossipNode;
        this.pendingPool = pendingPool;
        this.blockchain = blockchain;
        this.trainingNode = trainingNode;
        this.validatorNode = validatorNode;
        this.gson = new GsonBuilder().create();
    }

    /**
     * Triggered by AsyncRoundManager when local pool hits threshold.
     */
    public synchronized void initiateElection() {
        if (electionInProgress) return;
        electionInProgress = true;
        
        String poolHash = pendingPool.getPoolHash();
        System.out.println("[" + localNodeId + "] Initiating Leader Election for pool hash: " + poolHash);
        
        // Broadcast that we are ready to merge this specific pool hash
        GossipMessage msg = new GossipMessage(MessageType.READY_TO_MERGE, localNodeId, poolHash);
        gossipNode.broadcast(msg);
        
        // Add ourselves to the ready list
        handleReadyToMerge(localNodeId, poolHash);
        
        // Wait a short time for other nodes to broadcast their readiness
        ScheduledExecutorService scheduler = Executors.newSingleThreadScheduledExecutor();
        scheduler.schedule(this::resolveElection, 2, TimeUnit.SECONDS);
    }

    /**
     * Process incoming READY_TO_MERGE messages.
     */
    public synchronized void handleReadyToMerge(String nodeId, String poolHash) {
        readyNodes.computeIfAbsent(poolHash, k -> new ArrayList<>()).add(nodeId);
    }

    /**
     * Resolves the election. Lowest nodeId with the matching pool hash wins.
     */
    private synchronized void resolveElection() {
        String myPoolHash = pendingPool.getPoolHash();
        List<String> nodesWithMyHash = readyNodes.getOrDefault(myPoolHash, new ArrayList<>());
        
        if (nodesWithMyHash.isEmpty()) {
            electionInProgress = false;
            return; // Should never happen since we added ourselves
        }
        
        // Sort to find the lowest nodeId
        nodesWithMyHash.sort(String::compareTo);
        String leaderId = nodesWithMyHash.get(0);
        
        System.out.println("[" + localNodeId + "] Election resolved. Leader is: " + leaderId + " (out of " + nodesWithMyHash.size() + " nodes)");
        
        if (localNodeId.equals(leaderId)) {
            executeMergeAsLeader();
        }
        
        // Reset state
        readyNodes.clear();
        electionInProgress = false;
    }

    private void executeMergeAsLeader() {
        System.out.println("[" + localNodeId + "] I AM THE LEADER. Executing FedAvg...");
        
        List<Matrix[]> modelsToAverage = pendingPool.getUpdates();
        Matrix[] newGlobalWeights = FedAvg.aggregateWeights(modelsToAverage);
        
        // Create block
        // We use round = blockchain.size() as a simple proxy for the round number
        int round = blockchain.getChain().size() + 1;
        Block newBlock = new Block(localNodeId, round, "hash-placeholder", 0.0, System.currentTimeMillis(), blockchain.getLatestBlock().getHash());
        blockchain.addBlock(newBlock);
        
        // Broadcast MERGE_COMPLETE with the new weights
        String payload = gson.toJson(newGlobalWeights);
        GossipMessage msg = new GossipMessage(MessageType.MERGE_COMPLETE, localNodeId, payload);
        gossipNode.broadcast(msg);
        
        // Apply locally
        applyMergeComplete(newGlobalWeights);
    }

    /**
     * Called when MERGE_COMPLETE arrives (or when we generate it as leader).
     */
    public synchronized void applyMergeComplete(Matrix[] newGlobalWeights) {
        System.out.println("[" + localNodeId + "] Applying new global model...");
        pendingPool.clear();
        
        if (trainingNode != null) {
            trainingNode.onNewGlobalModel(newGlobalWeights);
        }
        if (validatorNode != null) {
            validatorNode.updateGlobalModel(newGlobalWeights);
        }
    }
}
