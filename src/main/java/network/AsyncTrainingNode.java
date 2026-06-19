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
    private Thread trainingThread;

    public AsyncTrainingNode(FederatedNode coreNode, GossipNode gossipNode, int epochsPerRound) {
        this.coreNode = coreNode;
        this.gossipNode = gossipNode;
        this.epochsPerRound = epochsPerRound;
        this.gson = new GsonBuilder().create();
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
                gossipNode.broadcast(msg);
                
                // Pause training until the next global model arrives to avoid spamming the network
                // with updates based on the exact same base model
                try {
                    synchronized (this) {
                        wait(); 
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
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
        notify(); // Wake up the training thread
    }
    
    public FederatedNode getCoreNode() {
        return coreNode;
    }
}
