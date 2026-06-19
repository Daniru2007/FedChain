package network;

/**
 * Payload for the HELLO gossip message.
 * Used for dynamic peer discovery.
 */
public class HelloPayload {
    public String ipAddress;
    public int port;

    public HelloPayload(String ipAddress, int port) {
        this.ipAddress = ipAddress;
        this.port = port;
    }
}
