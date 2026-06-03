package blockchain;

import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;

import static org.junit.jupiter.api.Assertions.*;

class BlockChainTest {

    @Test
    void addValidBlockAccepted() {
        BlockChain chain = new BlockChain(0.1);
        Block latest = chain.getLatestBlock();
        Block b = new Block("nodeA", 1, "m1", 0.05, System.currentTimeMillis(), latest.getHash());
        assertTrue(chain.addBlock(b));
        assertEquals(2, chain.getChain().size());
        assertTrue(chain.isChainValid());
    }

    @Test
    void rejectHighLoss() {
        BlockChain chain = new BlockChain(0.01);
        Block latest = chain.getLatestBlock();
        Block b = new Block("nodeA", 1, "m1", 0.5, System.currentTimeMillis(), latest.getHash());
        assertFalse(chain.addBlock(b));
        assertEquals(1, chain.getChain().size());
    }

    @Test
    void rejectPreviousHashMismatch() {
        BlockChain chain = new BlockChain(0.1);
        // wrong previous hash
        Block b = new Block("nodeA", 1, "m1", 0.01, System.currentTimeMillis(), "badprev");
        assertFalse(chain.addBlock(b));
        assertEquals(1, chain.getChain().size());
    }

    @Test
    void rejectTamperedBlockHash() throws Exception {
        BlockChain chain = new BlockChain(0.1);
        Block latest = chain.getLatestBlock();
        Block b = new Block("nodeA", 1, "m1", 0.01, System.currentTimeMillis(), latest.getHash());

        // tamper with the block's internal stored hash via reflection
        Field hashField = Block.class.getDeclaredField("hash");
        hashField.setAccessible(true);
        hashField.set(b, "0000deadbeef");

        assertFalse(chain.addBlock(b));
        assertEquals(1, chain.getChain().size());
    }
}

