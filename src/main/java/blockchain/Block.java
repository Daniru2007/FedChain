package blockchain;

import java.util.Objects;

public class Block {
	private final String nodeId;
	private final int round;
	private final String modelHash;
	private final double loss;
	private final long timestamp;
	private final String previousHash;
	private final String hash;

	public Block(String nodeId, int round, String modelHash, double loss, long timestamp, String previousHash) {
		this.nodeId = Objects.requireNonNull(nodeId, "nodeId cannot be null");
		this.round = round;
		this.modelHash = Objects.requireNonNull(modelHash, "modelHash cannot be null");
		this.loss = loss;
		this.timestamp = timestamp;
		this.previousHash = Objects.requireNonNull(previousHash, "previousHash cannot be null");
		String input = nodeId + round + modelHash + loss + timestamp + previousHash;
		this.hash = HashUtil.sha256(input);
	}

	public String getNodeId() {
		return nodeId;
	}

	public int getRound() {
		return round;
	}

	public String getModelHash() {
		return modelHash;
	}

	public double getLoss() {
		return loss;
	}

	public long getTimestamp() {
		return timestamp;
	}

	public String getPreviousHash() {
		return previousHash;
	}

	public String getHash() {
		return hash;
	}

	@Override
	public String toString() {
		return "Block{" +
				"nodeId='" + nodeId + '\'' +
				", round=" + round +
				", modelHash='" + modelHash + '\'' +
				", loss=" + loss +
				", timestamp=" + timestamp +
				", previousHash='" + previousHash + '\'' +
				", hash='" + hash + '\'' +
				'}';
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj) {
			return true;
		}
		if (!(obj instanceof Block other)) {
			return false;
		}
		return hash.equals(other.hash);
	}

	@Override
	public int hashCode() {
		return hash.hashCode();
	}
}
