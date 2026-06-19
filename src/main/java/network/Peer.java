package network;

import java.util.Objects;

public class Peer {
    private final String nodeId;
    private final String host;
    private final int port;
    private long lastSeen;
    private boolean reachable;

    public Peer(String nodeId, String host, int port) {
        this.nodeId = Objects.requireNonNull(nodeId);
        this.host = Objects.requireNonNull(host);
        this.port = port;
        this.lastSeen = System.currentTimeMillis();
        this.reachable = true;
    }

    public String getNodeId() {
        return nodeId;
    }

    public String getHost() {
        return host;
    }

    public int getPort() {
        return port;
    }

    public long getLastSeen() {
        return lastSeen;
    }

    public void setLastSeen(long lastSeen) {
        this.lastSeen = lastSeen;
    }

    public boolean isReachable() {
        return reachable;
    }

    public void setReachable(boolean reachable) {
        this.reachable = reachable;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Peer peer = (Peer) o;
        return port == peer.port && nodeId.equals(peer.nodeId) && host.equals(peer.host);
    }

    @Override
    public int hashCode() {
        return Objects.hash(nodeId, host, port);
    }
}
