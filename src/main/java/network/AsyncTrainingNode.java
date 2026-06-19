package network;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import math.Matrix;
import ml.FederatedNode;
import ml.NeuralNetwork;

/**
 * Wraps a FederatedNode to train asynchronously.
 * Trains continuously, gossips weights when done, and restarts when a new
 * global model arrives.
 */
public class AsyncTrainingNode {

    private final FederatedNode coreNode;
    private final GossipNode gossipNode;
    private final int epochsPerRound;
    private final Gson gson;
    
    private volatile boolean training = false;
    private volatile boolean modelUpdated = false;
    private Thread trainingThread;
    private ml.NeuralNetwork lastGlobalModel;

    public AsyncTrainingNode(FederatedNode coreNode, GossipNode gossipNode, int epochsPerRound) {
        this.coreNode = coreNode;
        this.gossipNode = gossipNode;
        this.epochsPerRound = epochsPerRound;
        this.gson = new GsonBuilder().create();
        this.lastGlobalModel = coreNode.getLocalModel(); // Initial pristine state
    }

    public void startTraining() {
        if (training) return;
        training = true;
        
        trainingThread = new Thread(() -> {
            while (training) {
                // Train locally
                System.out.println("[" + coreNode.getNodeId() + "] Started training...");
                coreNode.train(epochsPerRound);
                
                if (!training) break; // Interrupted by merge

                // Extract weights and broadcast
                Matrix[] weights = coreNode.getWeights();
                String payload = gson.toJson(weights);
                
                System.out.println("[" + coreNode.getNodeId() + "] Finished training. Gossiping WEIGHT_UPDATE.");
                GossipMessage msg = new GossipMessage(MessageType.WEIGHT_UPDATE, coreNode.getNodeId(), payload);
                
                // CRITICAL FIX: The node must process its own update locally to save it in its Pending Pool!
                gossipNode.processLocally(msg);
                gossipNode.broadcast(msg);
                
                // TRUE ASYNCHRONOUS FL:
                // We do NOT wait for the network to reach consensus!
                // We immediately loop back and start the next epoch on our current local model.
                // When the network eventually completes a merge, onNewGlobalModel() will be triggered
                // in the background, which will asynchronously overwrite our weights mid-stride!
                System.out.println("[" + coreNode.getNodeId() + "] Proceeding to next epoch continuously...");
            }
        });
        trainingThread.start();
    }

    public void stopTraining() {
        training = false;
        if (trainingThread != null) {
            trainingThread.interrupt();
        }
    }

    /**
     * Called when a MERGE_COMPLETE message arrives indicating a new global model.
     */
    public synchronized void onNewGlobalModel(Matrix[] newWeights) {
        System.out.println("[" + coreNode.getNodeId() + "] Received new global model. Updating and restarting training.");
        coreNode.setWeights(newWeights);
        lastGlobalModel = coreNode.getLocalModel(); // Save pristine state for the dashboard
        modelUpdated = true;
        notify(); // Wake up the training thread
    }
    
    public ml.NeuralNetwork getLatestGlobalModel() {
        return lastGlobalModel;
    }

    public FederatedNode getCoreNode() {
        return coreNode;
    }
}
