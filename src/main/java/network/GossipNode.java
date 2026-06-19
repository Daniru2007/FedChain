package network;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Consumer;

/**
 * The core networking component that handles P2P gossip via TCP.
 */
public class GossipNode {

    private static final int MAX_SEEN_MESSAGES = 10000;
    private static final int GOSSIP_FANOUT = 3;

    private final String nodeId;
    private final String modelId;
    private final String myIp;
    private final int port;
    private final PeerList peerList;
    private final Gson gson;

    // Callbacks for when a new message is received
    private Consumer<GossipMessage> messageListener;

    // LRU Cache for deduping seen messages
    private final Map<String, Boolean> seenMessages = Collections.synchronizedMap(
        new LinkedHashMap<String, Boolean>(MAX_SEEN_MESSAGES, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<String, Boolean> eldest) {
                return size() > MAX_SEEN_MESSAGES;
            }
        }
    );

    private ServerSocket serverSocket;
    private final ExecutorService serverPool = Executors.newCachedThreadPool();
    private final ExecutorService clientPool = Executors.newCachedThreadPool();
    private volatile boolean running = false;

    public GossipNode(String nodeId, int port, PeerList peerList) {
        this(nodeId, "mnist-run-001", "127.0.0.1", port, peerList);
    }

    public GossipNode(String nodeId, String modelId, String myIp, int port, PeerList peerList) {
        this.nodeId = nodeId;
        this.modelId = modelId;
        this.myIp = myIp;
        this.port = port;
        this.peerList = peerList;
        this.gson = new GsonBuilder().create();
    }

    public void setListener(Consumer<GossipMessage> listener) {
        this.messageListener = listener;
    }
    
    public void processLocally(GossipMessage msg) {
        if (messageListener != null) {
            messageListener.accept(msg);
        }
    }

    public void start() {
        if (running) return;
        running = true;

        serverPool.submit(() -> {
            try {
                serverSocket = new ServerSocket(port);
                System.out.println("[" + nodeId + "] Listening for gossip on port " + port);

                while (running) {
                    Socket clientSocket = serverSocket.accept();
                    serverPool.submit(() -> handleIncoming(clientSocket));
                }
            } catch (Exception e) {
                if (running) {
                    System.err.println("[" + nodeId + "] Server socket error: " + e.getMessage());
                }
            }
        });
    }

    public void stop() {
        running = false;
        try {
            if (serverSocket != null) serverSocket.close();
        } catch (Exception ignored) {}
        serverPool.shutdownNow();
        clientPool.shutdownNow();
    }

    private void handleIncoming(Socket socket) {
        try (
            socket;
            BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()))
        ) {
            String json = in.readLine();
            if (json == null || json.isEmpty()) return;

            GossipMessage msg = gson.fromJson(json, GossipMessage.class);

            // Network Segmentation: Ignore messages for other models
            if (!msg.getModelId().equals(this.modelId)) {
                return;
            }

            // Deduplication: if we've seen this message ID, ignore it (breaks gossip loops)
            if (seenMessages.putIfAbsent(msg.getMessageId(), true) != null) {
                return;
            }

            // If it's a HELLO message, parse the payload and add the new peer
            if (msg.getType() == MessageType.HELLO) {
                HelloPayload hello = gson.fromJson(msg.getPayload(), HelloPayload.class);
                Peer newPeer = new Peer(msg.getSenderId(), hello.ipAddress, hello.port);
                peerList.addPeer(newPeer);
                System.out.println("[" + nodeId + "] Discovered new peer: " + newPeer.getNodeId() + " (" + newPeer.getHost() + ":" + newPeer.getPort() + ")");
            }

            // Update peer last seen (the sender is alive)
            peerList.updateLastSeen(msg.getSenderId());

            // Process locally
            if (messageListener != null) {
                messageListener.accept(msg);
            }

            // Fanout to other peers
            forward(msg);

        } catch (Exception e) {
            // Ignore minor socket drops
        }
    }

    /**
     * Broadcasts a new message originating from this node.
     */
    public void broadcast(GossipMessage msg) {
        // Mark as seen so we don't process our own broadcast if echoed back
        seenMessages.put(msg.getMessageId(), true);
        forward(msg);
    }

    /**
     * Forwards a message to a random subset of peers.
     */
    private void forward(GossipMessage msg) {
        List<Peer> targets = peerList.getRandomPeers(GOSSIP_FANOUT);
        String json = gson.toJson(msg);

        for (Peer target : targets) {
            // Don't send it back to the original sender
            if (target.getNodeId().equals(msg.getSenderId())) continue;

            clientPool.submit(() -> sendToPeer(target, json));
        }
    }

    /**
     * Opens a stateless TCP connection to send the message.
     */
    private void sendToPeer(Peer peer, String json) {
        try (
            Socket socket = new Socket(peer.getHost(), peer.getPort());
            PrintWriter out = new PrintWriter(new OutputStreamWriter(socket.getOutputStream()), true)
        ) {
            socket.setSoTimeout(2000); // 2 second timeout
            out.println(json);
            peer.setReachable(true);
            peer.setLastSeen(System.currentTimeMillis());
        } catch (Exception e) {
            peer.setReachable(false);
        }
    }

    public String getNodeId() {
        return nodeId;
    }
}
