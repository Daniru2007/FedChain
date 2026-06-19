package network;

import blockchain.HashUtil;
import java.util.Objects;
import java.util.UUID;

/**
 * A generic envelope for any message sent across the gossip network.
 */
public class GossipMessage {
    private final String messageId;
    private final MessageType type;
    private final String senderId;
    private final long timestamp;
    private final String payload; // JSON serialized payload

    public GossipMessage(MessageType type, String senderId, String payload) {
        this.type = Objects.requireNonNull(type);
        this.senderId = Objects.requireNonNull(senderId);
        this.timestamp = System.currentTimeMillis();
        this.payload = payload != null ? payload : "";
        
        // Ensure global uniqueness using content + randomness so identical
        // broadcasts from different rounds get unique IDs
        this.messageId = HashUtil.sha256(senderId + timestamp + this.payload + UUID.randomUUID().toString());
    }

    public String getMessageId() {
        return messageId;
    }

    public MessageType getType() {
        return type;
    }

    public String getSenderId() {
        return senderId;
    }

    public long getTimestamp() {
        return timestamp;
    }

    public String getPayload() {
        return payload;
    }
}
