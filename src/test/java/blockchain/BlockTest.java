package blockchain;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class BlockTest {

    @Test
    void blockCreationWithValidInputs() {
        Block block = new Block("node-1", 1, "model-hash-abc", 0.05, 1717416000000L, "0");

        assertEquals("node-1", block.getNodeId());
        assertEquals(1, block.getRound());
        assertEquals("model-hash-abc", block.getModelHash());
        assertEquals(0.05, block.getLoss());
        assertEquals(1717416000000L, block.getTimestamp());
        assertEquals("0", block.getPreviousHash());
        assertEquals(64, block.getHash().length()); // SHA-256 hex is 64 chars
    }

    @Test
    void hashIsConsistent() {
        Block block1 = new Block("node-1", 1, "model-hash-abc", 0.05, 1717416000000L, "0");
        Block block2 = new Block("node-1", 1, "model-hash-abc", 0.05, 1717416000000L, "0");

        assertEquals(block1.getHash(), block2.getHash());
    }

    @Test
    void changingFieldChangesHash() {
        Block block1 = new Block("node-1", 1, "model-hash-abc", 0.05, 1717416000000L, "0");
        Block block2 = new Block("node-2", 1, "model-hash-abc", 0.05, 1717416000000L, "0");
        Block block3 = new Block("node-1", 2, "model-hash-abc", 0.05, 1717416000000L, "0");
        Block block4 = new Block("node-1", 1, "model-hash-xyz", 0.05, 1717416000000L, "0");
        Block block5 = new Block("node-1", 1, "model-hash-abc", 0.10, 1717416000000L, "0");

        assertNotEquals(block1.getHash(), block2.getHash(), "Changing nodeId should change hash");
        assertNotEquals(block1.getHash(), block3.getHash(), "Changing round should change hash");
        assertNotEquals(block1.getHash(), block4.getHash(), "Changing modelHash should change hash");
        assertNotEquals(block1.getHash(), block5.getHash(), "Changing loss should change hash");
    }

    @Test
    void blockEqualsAndHashCodeUsesHash() {
        Block block1 = new Block("node-1", 1, "model-hash-abc", 0.05, 1717416000000L, "0");
        Block block2 = new Block("node-1", 1, "model-hash-abc", 0.05, 1717416000000L, "0");
        Block block3 = new Block("node-1", 2, "model-hash-abc", 0.05, 1717416000000L, "0");

        assertEquals(block1, block2);
        assertEquals(block1.hashCode(), block2.hashCode());
        assertNotEquals(block1, block3);
    }

    @Test
    void nullFieldsAreRejected() {
        assertThrows(NullPointerException.class, () -> new Block(null, 1, "model-hash", 0.05, 1000L, "0"));
        assertThrows(NullPointerException.class, () -> new Block("node-1", 1, null, 0.05, 1000L, "0"));
        assertThrows(NullPointerException.class, () -> new Block("node-1", 1, "model-hash", 0.05, 1000L, null));
    }

    @Test
    void blockChainIntegrity() {
        Block block1 = new Block("node-1", 1, "model-1", 0.05, 1717416000000L, "0");
        Block block2 = new Block("node-1", 2, "model-2", 0.03, 1717416001000L, block1.getHash());
        Block block3 = new Block("node-1", 3, "model-3", 0.02, 1717416002000L, block2.getHash());

        assertEquals("0", block1.getPreviousHash());
        assertEquals(block1.getHash(), block2.getPreviousHash());
        assertEquals(block2.getHash(), block3.getPreviousHash());
    }
}

