package network;

import blockchain.Block;
import blockchain.BlockChain;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;
import math.Matrix;
import java.util.List;

/**
 * Ensures this node's blockchain is synchronized with the network.
 * Follows the longest-valid-chain-wins rule.
 */
public class ChainSync {

    private final String localNodeId;
    private final GossipNode gossipNode;
    private final BlockChain localBlockchain;
    private final Gson gson;
    
    // Optional: Only trainers or validators need to be notified when a new
    // longer chain is accepted so they can update their global model.
    private final LeaderElection leaderElection; 

    public ChainSync(String localNodeId, GossipNode gossipNode, BlockChain localBlockchain, LeaderElection leaderElection) {
        this.localNodeId = localNodeId;
        this.gossipNode = gossipNode;
        this.localBlockchain = localBlockchain;
        this.leaderElection = leaderElection;
        this.gson = new GsonBuilder().create();
    }

    /**
     * Call this periodically or on startup to request chains from peers.
     */
    public void requestChains() {
        GossipMessage msg = new GossipMessage(MessageType.CHAIN_REQUEST, localNodeId, "");
        gossipNode.broadcast(msg);
    }

    /**
     * When another node requests our chain, we broadcast it back.
     */
    public void handleChainRequest(GossipMessage msg) {
        String payload = gson.toJson(localBlockchain.getChain());
        GossipMessage response = new GossipMessage(MessageType.CHAIN_RESPONSE, localNodeId, payload);
        gossipNode.broadcast(response);
    }

    /**
     * When we receive a chain from another node, evaluate if it's longer and valid.
     */
    public synchronized void handleChainResponse(GossipMessage msg) {
        List<Block> incomingChain = gson.fromJson(msg.getPayload(), new TypeToken<List<Block>>(){}.getType());
        
        if (incomingChain == null || incomingChain.isEmpty()) return;

        // Longest-chain rule
        if (incomingChain.size() > localBlockchain.getChain().size()) {
            
            // Temporary blockchain to validate the incoming chain
            BlockChain temp = new BlockChain(10.0); // Threshold doesn't matter for this validation check
            // Override the genesis block to match incoming so we can check hashes
            temp.getChain().clear();
            
            boolean isValid = true;
            for (Block b : incomingChain) {
                temp.getChain().add(b);
            }
            
            if (temp.isChainValid()) {
                System.out.println("[" + localNodeId + "] Found a longer valid chain (length " + incomingChain.size() + "). Syncing...");
                localBlockchain.getChain().clear();
                localBlockchain.getChain().addAll(incomingChain);
                
                // In a full implementation, we'd extract the actual weights from the
                // latest block if it contained them. For our simulation, the weights 
                // are transmitted via MERGE_COMPLETE. ChainSync guarantees historical agreement.
            }
        }
    }
}
