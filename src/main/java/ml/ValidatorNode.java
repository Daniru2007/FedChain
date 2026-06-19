package ml;

import math.Matrix;

import java.util.Objects;

/**
 * An independent validator that evaluates submitted model weights against a
 * shared validation dataset and casts a {@link Vote}.
 *
 * <p>A submitted update is accepted if its validation accuracy is at least
 * {@code (currentGlobalAccuracy - tolerance)}. This relative threshold avoids
 * false rejections in early rounds when overall accuracy is still low.
 *
 * <p>The validator never trains — it only evaluates.
 */
public class ValidatorNode {

    private final String validatorId;
    private final Matrix[] validationImages;
    private final Matrix[] validationLabels;
    /** Accuracy must not fall below (globalAccuracy - toleranceMargin) to be accepted. */
    private final double toleranceMargin;

    /**
     * @param validatorId      unique name for this validator (for logging)
     * @param validationImages shared validation image set (column-vector Matrices)
     * @param validationLabels one-hot label vectors matching validationImages
     * @param toleranceMargin  how many accuracy points below the global model are tolerated (e.g. 0.05 = 5%)
     */
    public ValidatorNode(String validatorId,
                         Matrix[] validationImages,
                         Matrix[] validationLabels,
                         double toleranceMargin) {
        this.validatorId       = Objects.requireNonNull(validatorId, "validatorId cannot be null");
        this.validationImages  = Objects.requireNonNull(validationImages, "validationImages cannot be null");
        this.validationLabels  = Objects.requireNonNull(validationLabels, "validationLabels cannot be null");
        if (validationImages.length != validationLabels.length) {
            throw new IllegalArgumentException("validationImages and validationLabels must be the same length.");
        }
        if (toleranceMargin < 0 || toleranceMargin > 1) {
            throw new IllegalArgumentException("toleranceMargin must be in [0, 1].");
        }
        this.toleranceMargin = toleranceMargin;
    }

    /**
     * Evaluates {@code candidateWeights} on the validation set.
     *
     * @param candidateWeights  the weight/bias parameters submitted by a training node
     * @param referenceModel    the current global model — used to compute the baseline accuracy
     * @return {@link Vote#ACCEPT} if the candidate meets the threshold, {@link Vote#REJECT} otherwise
     */
    public Vote evaluate(Matrix[] candidateWeights, NeuralNetwork referenceModel) {
        Objects.requireNonNull(candidateWeights, "candidateWeights cannot be null");
        Objects.requireNonNull(referenceModel, "referenceModel cannot be null");

        // Measure global model baseline
        double globalAccuracy = Evaluator.evaluate(referenceModel, validationImages, validationLabels);

        // Inject candidate weights into a temporary model and measure its accuracy
        NeuralNetwork candidate = referenceModel.copy();
        candidate.setParameters(candidateWeights);
        double candidateAccuracy = Evaluator.evaluate(candidate, validationImages, validationLabels);

        double threshold = globalAccuracy - toleranceMargin;
        Vote vote = (candidateAccuracy >= threshold) ? Vote.ACCEPT : Vote.REJECT;

        System.out.printf("    [%s] global=%.2f%% candidate=%.2f%% threshold=%.2f%% → %s%n",
                validatorId,
                globalAccuracy * 100.0,
                candidateAccuracy * 100.0,
                threshold * 100.0,
                vote);

        return vote;
    }

    public String getValidatorId() {
        return validatorId;
    }
}
