package blockchain;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class BlockChain {
	private final List<Block> chain = new ArrayList<>();
	private final double lossThreshold;

	public BlockChain(double lossThreshold) {
		this.lossThreshold = lossThreshold;
		// create genesis block
		chain.add(createGenesisBlock());
	}

	private Block createGenesisBlock() {
		long now = System.currentTimeMillis();
		return new Block("GENESIS", 0, "0", 0.0, now, "0000000000");
	}

	public Block getLatestBlock() {
		return chain.get(chain.size() - 1);
	}

	public boolean addBlock(Block block) {
		// 1. loss below threshold
		if (block.getLoss() > lossThreshold) {
			return false;
		}
		// 2. previousHash matches latest block's hash
		if (!block.getPreviousHash().equals(getLatestBlock().getHash())) {
			return false;
		}
		// 3. block's own hash is valid
		if (!isBlockValid(block)) {
			return false;
		}

		chain.add(block);
		return true;
	}

	public boolean isBlockValid(Block block) {
		String input = block.getNodeId() + block.getRound() + block.getModelHash() + block.getLoss() + block.getTimestamp() + block.getPreviousHash();
		String recalculated = HashUtil.sha256(input);
		return recalculated.equals(block.getHash());
	}

	public boolean isChainValid() {
		for (int i = 0; i < chain.size(); i++) {
			Block current = chain.get(i);
			// validate current's hash
			if (!isBlockValid(current)) return false;
			// check linkage
			if (i > 0) {
				Block previous = chain.get(i - 1);
				if (!current.getPreviousHash().equals(previous.getHash())) return false;
			}
		}
		return true;
	}

	public List<Block> getChain() {
		return Collections.unmodifiableList(chain);
	}

}
