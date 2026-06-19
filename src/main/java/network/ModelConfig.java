package network;

public class ModelConfig {
    public String modelId;
    public int[] architecture;
    public double learningRate;
    public int mergeThreshold;
    public int requiredValidators;
    public double validatorTolerance;

    public ModelConfig(String modelId, int[] architecture, double learningRate, 
                       int mergeThreshold, int requiredValidators, double validatorTolerance) {
        this.modelId = modelId;
        this.architecture = architecture;
        this.learningRate = learningRate;
        this.mergeThreshold = mergeThreshold;
        this.requiredValidators = requiredValidators;
        this.validatorTolerance = validatorTolerance;
    }
}
