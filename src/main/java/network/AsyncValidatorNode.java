package network;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import math.Matrix;
import ml.NeuralNetwork;
import ml.ValidatorNode;
import ml.Vote;

import java.util.Objects;

/**
 * Payload structure for a VOTE message.
 */
class VotePayload {
    String updateId;
    boolean isAccept;
    double accuracy;

    public VotePayload(String updateId, boolean isAccept, double accuracy) {
        this.updateId = updateId;
        this.isAccept = isAccept;
        this.accuracy = accuracy;
    }
}

/**
 * Wraps a ValidatorNode to run asynchronously.
 * Listens for WEIGHT_UPDATE messages, evaluates them, and gossips a VOTE.
 */
public class AsyncValidatorNode {

    private final ValidatorNode coreValidator;
    private final GossipNode gossipNode;
    private final Gson gson;
    
    // The baseline global model to compare updates against
    private NeuralNetwork currentGlobalModel;

    public AsyncValidatorNode(ValidatorNode coreValidator, GossipNode gossipNode, NeuralNetwork initialGlobalModel) {
        this.coreValidator = coreValidator;
        this.gossipNode = gossipNode;
        this.currentGlobalModel = Objects.requireNonNull(initialGlobalModel).copy();
        this.gson = new GsonBuilder().create();
    }
    
    public NeuralNetwork getCurrentGlobalModel() {
        return currentGlobalModel;
    }

    /**
     * Updates the baseline model when a MERGE_COMPLETE occurs.
     */
    public void updateGlobalModel(Matrix[] newWeights) {
        currentGlobalModel.setParameters(newWeights);
    }

    /**
     * Evaluates a candidate update and broadcasts the vote.
     */
    public void processWeightUpdate(GossipMessage msg) {
        if (msg.getType() != MessageType.WEIGHT_UPDATE) return;
        
        Matrix[] candidateWeights = gson.fromJson(msg.getPayload(), Matrix[].class);
        
        System.out.println("[" + coreValidator.getValidatorId() + "] Evaluating update from " + msg.getSenderId());
        
        Vote vote = coreValidator.evaluate(candidateWeights, currentGlobalModel);
        boolean isAccept = (vote == Vote.ACCEPT);
        
        // Broadcast the vote
        VotePayload vPayload = new VotePayload(msg.getMessageId(), isAccept, 0.0); // Evaluator prints accuracy internally for now
        String payloadJson = gson.toJson(vPayload);
        
        GossipMessage voteMsg = new GossipMessage(MessageType.VOTE, coreValidator.getValidatorId(), payloadJson);
        gossipNode.broadcast(voteMsg);
    }
}
