package network;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.stream.Collectors;

public class PeerList {
    private final List<Peer> peers = new CopyOnWriteArrayList<>();

    public void addPeer(Peer peer) {
        if (!peers.contains(peer)) {
            peers.add(peer);
        }
    }

    public void removePeer(String nodeId) {
        peers.removeIf(p -> p.getNodeId().equals(nodeId));
    }

    public Optional<Peer> getPeer(String nodeId) {
        return peers.stream().filter(p -> p.getNodeId().equals(nodeId)).findFirst();
    }

    public void updateLastSeen(String nodeId) {
        getPeer(nodeId).ifPresent(p -> {
            p.setLastSeen(System.currentTimeMillis());
            p.setReachable(true);
        });
    }

    public List<Peer> getAllPeers() {
        return new ArrayList<>(peers);
    }

    /**
     * Returns a random subset of known reachable peers for gossip fanout.
     */
    public List<Peer> getRandomPeers(int count) {
        List<Peer> reachablePeers = peers.stream()
                .filter(Peer::isReachable)
                .collect(Collectors.toList());
        
        Collections.shuffle(reachablePeers);
        return reachablePeers.subList(0, Math.min(count, reachablePeers.size()));
    }
}
