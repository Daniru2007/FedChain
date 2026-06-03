package ml;

public final class ActivationFunction {
    private ActivationFunction() {
    }

    public static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public static double sigmoidDerivative(double x) {
        double sigmoid = sigmoid(x);
        return sigmoid * (1.0 - sigmoid);
    }

    public static double relu(double x) {
        return Math.max(0.0, x);
    }
}
