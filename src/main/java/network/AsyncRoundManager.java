package network;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import math.Matrix;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Consumer;

/**
 * The central brain of an async node. Listens to gossip, tracks votes,
 * manages the pending pool, and triggers leader elections.
 */
public class AsyncRoundManager implements Consumer<GossipMessage> {

    private final String localNodeId;
    private final PendingPool pendingPool;
    private final VoteTracker voteTracker;
    private final LeaderElection leaderElection;
    private final AsyncValidatorNode validatorNode;
    private final Gson gson;
    
    private final int requiredValidators;
    private final int mergeThreshold;
    private ChainSync chainSync;

    // Caches weights so when enough votes arrive, we can add them to the pool
    // updateId -> Matrix[]
    private final Map<String, Matrix[]> updateCache = new ConcurrentHashMap<>();

    public AsyncRoundManager(String localNodeId, PendingPool pendingPool, VoteTracker voteTracker, 
                             LeaderElection leaderElection, AsyncValidatorNode validatorNode,
                             int requiredValidators, int mergeThreshold) {
        this.localNodeId = localNodeId;
        this.pendingPool = pendingPool;
        this.voteTracker = voteTracker;
        this.leaderElection = leaderElection;
        this.validatorNode = validatorNode;
        this.requiredValidators = requiredValidators;
        this.mergeThreshold = mergeThreshold;
        this.gson = new GsonBuilder().create();
    }

    public void setChainSync(ChainSync chainSync) {
        this.chainSync = chainSync;
    }

    @Override
    public void accept(GossipMessage msg) {
        switch (msg.getType()) {
            case WEIGHT_UPDATE:
                handleWeightUpdate(msg);
                break;
            case VOTE:
                handleVote(msg);
                break;
            case READY_TO_MERGE:
                leaderElection.handleReadyToMerge(msg.getSenderId(), msg.getPayload());
                break;
            case MERGE_COMPLETE:
                Matrix[] newWeights = gson.fromJson(msg.getPayload(), Matrix[].class);
                voteTracker.clearAll();
                updateCache.clear();
                leaderElection.applyMergeComplete(newWeights);
                break;
            case CHAIN_REQUEST:
                if (chainSync != null) chainSync.handleChainRequest(msg);
                break;
            case CHAIN_RESPONSE:
                if (chainSync != null) chainSync.handleChainResponse(msg);
                break;
            default:
                break;
        }
    }

    private void handleWeightUpdate(GossipMessage msg) {
        Matrix[] weights = gson.fromJson(msg.getPayload(), Matrix[].class);
        updateCache.put(msg.getMessageId(), weights);
        
        // If this node is a validator, it should evaluate it and broadcast a vote
        if (validatorNode != null) {
            validatorNode.processWeightUpdate(msg);
        }
    }

    private void handleVote(GossipMessage msg) {
        VotePayload vote = gson.fromJson(msg.getPayload(), VotePayload.class);
        voteTracker.addVote(vote.updateId, msg.getSenderId(), vote.isAccept);
        
        // Did it cross the threshold?
        if (vote.isAccept && voteTracker.getAcceptCount(vote.updateId) >= requiredValidators) {
            Matrix[] weights = updateCache.get(vote.updateId);
            if (weights != null) {
                // Find the original sender of the update
                // (For simplicity we might just use the updateId as the key if we lost the sender ID,
                // but PendingPool just needs a unique string key per contribution to prevent duplicates)
                pendingPool.addUpdate(vote.updateId, weights);
                updateCache.remove(vote.updateId); // Free memory
                
                System.out.println("[" + localNodeId + "] Update " + vote.updateId.substring(0, 8) + " reached consensus. Pool size: " + pendingPool.size() + "/" + mergeThreshold);
                
                if (pendingPool.size() >= mergeThreshold) {
                    leaderElection.initiateElection();
                }
            }
        }
    }
}
