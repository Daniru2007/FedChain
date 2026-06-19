package ml;

import math.Matrix;

import java.util.List;
import java.util.Objects;

/**
 * Decentralized consensus engine that broadcasts a submitted model update to
 * all registered {@link ValidatorNode}s and aggregates their votes.
 *
 * <p>Decision rule: simple majority — more {@link Vote#ACCEPT}s than
 * {@link Vote#REJECT}s → the update is legitimate.
 *
 * <p>The engine is stateless between rounds; it acts purely as a routing and
 * counting layer between training nodes and validators.
 */
public class ConsensusEngine {

    private final List<ValidatorNode> validators;

    /**
     * @param validators the set of independent validators; must not be empty
     */
    public ConsensusEngine(List<ValidatorNode> validators) {
        Objects.requireNonNull(validators, "validators cannot be null");
        if (validators.isEmpty()) {
            throw new IllegalArgumentException("At least one validator is required.");
        }
        this.validators = List.copyOf(validators);
    }

    /**
     * Runs the consensus protocol for a single candidate update.
     *
     * <p>The candidate weights are broadcast to every validator, which each
     * independently casts a vote. A simple majority of accepts is required
     * for the update to be approved.
     *
     * @param nodeId           the ID of the submitting training node (for logging)
     * @param candidateWeights the model weights proposed by the training node
     * @param globalModel      the current global model used as baseline by validators
     * @return {@code true} if the update passed consensus, {@code false} if rejected
     */
    public boolean reachConsensus(String nodeId, Matrix[] candidateWeights, NeuralNetwork globalModel) {
        Objects.requireNonNull(nodeId, "nodeId cannot be null");
        Objects.requireNonNull(candidateWeights, "candidateWeights cannot be null");
        Objects.requireNonNull(globalModel, "globalModel cannot be null");

        System.out.printf("  [ConsensusEngine] Broadcasting update from %s to %d validators:%n",
                nodeId, validators.size());

        int accepts = 0;
        int rejects = 0;

        for (ValidatorNode validator : validators) {
            Vote vote = validator.evaluate(candidateWeights, globalModel);
            if (vote == Vote.ACCEPT) accepts++;
            else                     rejects++;
        }

        boolean passed = accepts > rejects;
        System.out.printf("  [ConsensusEngine] Result for %s: %d accept(s), %d reject(s) → %s%n",
                nodeId, accepts, rejects, passed ? "PASSED" : "FAILED");
        return passed;
    }

    public int validatorCount() {
        return validators.size();
    }
}
