package network;

import java.util.Collections;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Tracks incoming VOTE messages for each update.
 * When an update reaches the required majority, it can be marked as passed.
 */
public class VoteTracker {
    
    // updateId -> Set of validator nodeIds that voted ACCEPT
    private final Map<String, Set<String>> accepts = new ConcurrentHashMap<>();
    
    // updateId -> Set of validator nodeIds that voted REJECT
    private final Map<String, Set<String>> rejects = new ConcurrentHashMap<>();

    public void addVote(String updateId, String validatorId, boolean isAccept) {
        if (isAccept) {
            accepts.computeIfAbsent(updateId, k -> Collections.synchronizedSet(new HashSet<>())).add(validatorId);
        } else {
            rejects.computeIfAbsent(updateId, k -> Collections.synchronizedSet(new HashSet<>())).add(validatorId);
        }
    }

    public int getAcceptCount(String updateId) {
        Set<String> s = accepts.get(updateId);
        return s == null ? 0 : s.size();
    }

    public int getRejectCount(String updateId) {
        Set<String> s = rejects.get(updateId);
        return s == null ? 0 : s.size();
    }
    
    public void clear(String updateId) {
        accepts.remove(updateId);
        rejects.remove(updateId);
    }
    
    public void clearAll() {
        accepts.clear();
        rejects.clear();
    }
}
