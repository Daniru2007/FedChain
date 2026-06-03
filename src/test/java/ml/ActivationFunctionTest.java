package ml;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ActivationFunctionTest {

    @Test
    void sigmoidAndDerivativeMatchKnownValues() {
        assertEquals(0.5, ActivationFunction.sigmoid(0.0), 1e-12);
        assertEquals(0.25, ActivationFunction.sigmoidDerivative(0.0), 1e-12);
    }

    @Test
    void reluReturnsZeroForNegativeValues() {
        assertEquals(0.0, ActivationFunction.relu(-3.5), 1e-12);
        assertEquals(2.5, ActivationFunction.relu(2.5), 1e-12);
    }
}
