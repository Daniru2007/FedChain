package math;

import ml.ActivationFunction;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

class MatrixTest {

    @Test
    void constructorCopiesInputAndExposesDimensions() {
        double[][] values = {
                {1.0, 2.0},
                {3.0, 4.0}
        };

        Matrix matrix = new Matrix(values);

        assertEquals(2, matrix.getRows());
        assertEquals(2, matrix.getCols());
        assertEquals(1.0, matrix.get(0, 0));
        assertEquals(4.0, matrix.get(1, 1));

        double[][] copy = matrix.getData();
        assertNotSame(values, copy);
        assertArrayEquals(values[0], copy[0]);
        assertArrayEquals(values[1], copy[1]);

        values[0][0] = 99.0;
        assertEquals(1.0, matrix.get(0, 0));
    }

    @Test
    void addSubtractAndElementWiseMultiplyWorkForSameSizedMatrices() {
        Matrix a = new Matrix(new double[][]{
                {1.0, 2.0},
                {3.0, 4.0}
        });
        Matrix b = new Matrix(new double[][]{
                {5.0, 6.0},
                {7.0, 8.0}
        });

        Matrix sum = a.add(b);
        Matrix difference = b.subtract(a);
        Matrix product = a.elementWiseMultiply(b);

        assertArrayEquals(new double[]{6.0, 8.0}, sum.getData()[0]);
        assertArrayEquals(new double[]{10.0, 12.0}, sum.getData()[1]);
        assertArrayEquals(new double[]{4.0, 4.0}, difference.getData()[0]);
        assertArrayEquals(new double[]{4.0, 4.0}, difference.getData()[1]);
        assertArrayEquals(new double[]{5.0, 12.0}, product.getData()[0]);
        assertArrayEquals(new double[]{21.0, 32.0}, product.getData()[1]);
    }

    @Test
    void multiplyTransposeScaleAndApplyFunctionWork() {
        Matrix a = new Matrix(new double[][]{
                {1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0}
        });
        Matrix b = new Matrix(new double[][]{
                {7.0, 8.0},
                {9.0, 10.0},
                {11.0, 12.0}
        });

        Matrix multiplied = a.multiply(b);
        Matrix transposed = a.transpose();
        Matrix scaled = a.scale(2.0);
        Matrix sigmoid = a.applyFunction(ActivationFunction::sigmoid);

        assertArrayEquals(new double[]{58.0, 64.0}, multiplied.getData()[0]);
        assertArrayEquals(new double[]{139.0, 154.0}, multiplied.getData()[1]);
        assertArrayEquals(new double[]{1.0, 4.0}, transposed.getData()[0]);
        assertArrayEquals(new double[]{2.0, 5.0}, transposed.getData()[1]);
        assertArrayEquals(new double[]{3.0, 6.0}, transposed.getData()[2]);
        assertArrayEquals(new double[]{2.0, 4.0, 6.0}, scaled.getData()[0]);
        assertArrayEquals(new double[]{8.0, 10.0, 12.0}, scaled.getData()[1]);
        assertEquals(0.7310585786, sigmoid.get(0, 0), 1e-10);
    }

    @Test
    void incompatibleOperationsThrowHelpfulErrors() {
        Matrix a = new Matrix(new double[][]{
                {1.0, 2.0},
                {3.0, 4.0}
        });
        Matrix b = new Matrix(new double[][]{
                {5.0, 6.0, 7.0}
        });

        assertThrows(IllegalArgumentException.class, () -> a.add(b));
        assertThrows(IllegalArgumentException.class, () -> a.subtract(b));
        assertThrows(IllegalArgumentException.class, () -> a.elementWiseMultiply(b));
        assertThrows(IllegalArgumentException.class, () -> a.multiply(b));
    }

    @Test
    void invalidConstructionAndNullFunctionAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> new Matrix(-1, 1));
        assertThrows(IllegalArgumentException.class, () -> new Matrix(new double[][]{
                {1.0, 2.0},
                {3.0}
        }));
        assertThrows(NullPointerException.class, () -> new Matrix((double[][]) null));

        Matrix matrix = new Matrix(new double[][]{{1.0}});
        assertThrows(NullPointerException.class, () -> matrix.applyFunction(null));
    }

    @Test
    void equalsAndHashCodeCompareMatrixContents() {
        Matrix a = new Matrix(new double[][]{{1.0, 2.0}});
        Matrix b = new Matrix(new double[][]{{1.0, 2.0}});
        Matrix c = new Matrix(new double[][]{{2.0, 3.0}});

        assertEquals(a, b);
        assertEquals(a.hashCode(), b.hashCode());
        assertNotSame(a, b);
        assertFalse(a.equals(c));
    }
}
