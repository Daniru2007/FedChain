package network;

import blockchain.BlockChain;
import math.Matrix;
import ml.FederatedNode;
import ml.NeuralNetwork;
import ml.ValidatorNode;

import java.util.List;

/**
 * Encapsulates a fully wired P2P node.
 * Connects the ML, Blockchain, and Networking components together.
 */
public class NodeLauncher {

    public enum Role {
        TRAINER,
        VALIDATOR
    }

    private final String nodeId;
    private final Role role;
    private final GossipNode gossipNode;
    private final AsyncTrainingNode trainingNode;
    private final AsyncValidatorNode validatorNode;
    private final ChainSync chainSync;

    public NodeLauncher(String nodeId, int port, Role role, List<Peer> initialPeers, 
                        NeuralNetwork globalModel, BlockChain initialBlockchain,
                        Matrix[] localImages, Matrix[] localLabels, 
                        int mergeThreshold, int requiredValidators) {
        
        this.nodeId = nodeId;
        this.role = role;
        
        // Networking core
        PeerList peerList = new PeerList();
        initialPeers.forEach(peerList::addPeer);
        this.gossipNode = new GossipNode(nodeId, port, peerList);

        // Core data structures
        PendingPool pendingPool = new PendingPool();
        VoteTracker voteTracker = new VoteTracker();
        
        // Role-specific ML components
        if (role == Role.TRAINER) {
            FederatedNode coreNode = new FederatedNode(nodeId, globalModel, localImages, localLabels);
            this.trainingNode = new AsyncTrainingNode(coreNode, gossipNode, 1); // 1 epoch per cycle
            this.validatorNode = null;
        } else {
            ValidatorNode coreValidator = new ValidatorNode(nodeId, localImages, localLabels, 1.0); // 1.0 tolerance for testing
            this.validatorNode = new AsyncValidatorNode(coreValidator, gossipNode, globalModel);
            this.trainingNode = null;
        }

        // Distributed Coordination
        LeaderElection leaderElection = new LeaderElection(
            nodeId, gossipNode, pendingPool, initialBlockchain, trainingNode, validatorNode
        );

        AsyncRoundManager roundManager = new AsyncRoundManager(
            nodeId, pendingPool, voteTracker, leaderElection, validatorNode, 
            requiredValidators, mergeThreshold
        );

        this.chainSync = new ChainSync(nodeId, gossipNode, initialBlockchain, leaderElection);
        roundManager.setChainSync(chainSync);

        // Wire listener
        gossipNode.setListener(roundManager);
    }

    public void start() {
        // Start TCP server
        gossipNode.start();
        
        // Sync chain to make sure we have latest
        chainSync.requestChains();

        // Start async training if applicable
        if (role == Role.TRAINER) {
            trainingNode.startTraining();
        }
        
        System.out.println(">>> Node " + nodeId + " (" + role + ") started and listening on port " + gossipNode.getNodeId());
    }

    public void stop() {
        if (trainingNode != null) trainingNode.stopTraining();
        gossipNode.stop();
    }
    
    public AsyncTrainingNode getTrainingNode() {
        return trainingNode;
    }

    public AsyncValidatorNode getValidatorNode() {
        return validatorNode;
    }
}
