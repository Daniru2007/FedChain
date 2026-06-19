package network;

import blockchain.HashUtil;
import math.Matrix;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Holds updates that have passed consensus but haven't been merged yet.
 */
public class PendingPool {
    
    // trainerNodeId -> accepted weights
    private final Map<String, Matrix[]> pool = new ConcurrentHashMap<>();

    public void addUpdate(String trainerNodeId, Matrix[] weights) {
        pool.put(trainerNodeId, weights);
    }

    public int size() {
        return pool.size();
    }

    public void clear() {
        pool.clear();
    }

    public List<Matrix[]> getUpdates() {
        return new ArrayList<>(pool.values());
    }

    public List<String> getContributors() {
        return new ArrayList<>(pool.keySet());
    }

    /**
     * Critical for leader election: generates a hash of the current pool contents.
     * Only nodes with matching hashes can participate in the same election.
     */
    public String getPoolHash() {
        if (pool.isEmpty()) return "empty";
        
        List<String> sortedKeys = new ArrayList<>(pool.keySet());
        sortedKeys.sort(String::compareTo);
        
        StringBuilder sb = new StringBuilder();
        for (String key : sortedKeys) {
            sb.append(key).append("|");
        }
        return HashUtil.sha256(sb.toString());
    }
}
