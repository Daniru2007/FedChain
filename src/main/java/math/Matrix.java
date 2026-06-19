package math;

import java.util.Arrays;
import java.util.Objects;
import java.util.function.DoubleUnaryOperator;

public class Matrix {
	private final double[][] data;
	private final int rows;
	private final int cols;

	public Matrix(int rows, int cols) {
		if (rows < 0 || cols < 0) {
			throw new IllegalArgumentException("Matrix dimensions cannot be negative.");
		}
		this.rows = rows;
		this.cols = cols;
		this.data = new double[rows][cols];
	}

	public Matrix(double[][] values) {
		Objects.requireNonNull(values, "Matrix values cannot be null.");
		if (values.length == 0) {
			this.rows = 0;
			this.cols = 0;
			this.data = new double[0][0];
			return;
		}
		if (values[0] == null) {
			throw new IllegalArgumentException("Matrix rows cannot be null.");
		}

		this.rows = values.length;
		this.cols = values[0].length;
		this.data = new double[rows][cols];

		for (int i = 0; i < rows; i++) {
			if (values[i] == null) {
				throw new IllegalArgumentException("Matrix rows cannot be null.");
			}
			if (values[i].length != cols) {
				throw new IllegalArgumentException("Matrix must be rectangular.");
			}
			System.arraycopy(values[i], 0, this.data[i], 0, cols);
		}
	}

	public int getRows() {
		return rows;
	}

	public int getCols() {
		return cols;
	}

	public double get(int row, int col) {
		validateIndex(row, col);
		return data[row][col];
	}

	public double[][] getData() {
		double[][] copy = new double[rows][cols];
		for (int i = 0; i < rows; i++) {
			System.arraycopy(data[i], 0, copy[i], 0, cols);
		}
		return copy;
	}

	public Matrix add(Matrix b) {
		requireSameSize(b, "add");
		double[][] result = new double[rows][cols];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				result[i][j] = data[i][j] + b.data[i][j];
			}
		}
		return new Matrix(result);
	}

	public Matrix subtract(Matrix b) {
		requireSameSize(b, "subtract");
		double[][] result = new double[rows][cols];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				result[i][j] = data[i][j] - b.data[i][j];
			}
		}
		return new Matrix(result);
	}

	public Matrix multiply(Matrix b) {
		Objects.requireNonNull(b, "Matrix to multiply cannot be null.");
		if (cols != b.rows) {
			throw new IllegalArgumentException("Matrix dimensions are incompatible for multiplication.");
		}

		double[][] result = new double[rows][b.cols];
		if (b.cols == 1) {
			// Column vector optimization to avoid inner-loop 2D array pointer dereferencing
			double[] bCol = new double[cols];
			for (int k = 0; k < cols; k++) {
				bCol[k] = b.data[k][0];
			}
			for (int i = 0; i < rows; i++) {
				double sum = 0.0;
				double[] rowData = data[i];
				for (int k = 0; k < cols; k++) {
					sum += rowData[k] * bCol[k];
				}
				result[i][0] = sum;
			}
		} else {
			// Cache-friendly general matrix multiplication
			for (int i = 0; i < rows; i++) {
				double[] rowA = data[i];
				double[] rowRes = result[i];
				for (int k = 0; k < cols; k++) {
					double valA = rowA[k];
					double[] rowB = b.data[k];
					for (int j = 0; j < b.cols; j++) {
						rowRes[j] += valA * rowB[j];
					}
				}
			}
		}
		return new Matrix(result);
	}


	public Matrix transpose() {
		double[][] result = new double[cols][rows];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				result[j][i] = data[i][j];
			}
		}
		return new Matrix(result);
	}

	public Matrix scale(double value) {
		double[][] result = new double[rows][cols];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				result[i][j] = data[i][j] * value;
			}
		}
		return new Matrix(result);
	}

	public Matrix applyFunction(DoubleUnaryOperator operator) {
		Objects.requireNonNull(operator, "Function cannot be null.");
		double[][] result = new double[rows][cols];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				result[i][j] = operator.applyAsDouble(data[i][j]);
			}
		}
		return new Matrix(result);
	}

	public Matrix elementWiseMultiply(Matrix b) {
		requireSameSize(b, "element-wise multiply");
		double[][] result = new double[rows][cols];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				result[i][j] = data[i][j] * b.data[i][j];
			}
		}
		return new Matrix(result);
	}

	private void requireSameSize(Matrix other, String operation) {
		Objects.requireNonNull(other, "Matrix to " + operation + " cannot be null.");
		if (rows != other.rows || cols != other.cols) {
			throw new IllegalArgumentException("Matrix dimensions must match to " + operation + ".");
		}
	}

	private void validateIndex(int row, int col) {
		if (row < 0 || row >= rows || col < 0 || col >= cols) {
			throw new IndexOutOfBoundsException("Index out of bounds: (" + row + ", " + col + ")");
		}
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj) {
			return true;
		}
		if (!(obj instanceof Matrix other)) {
			return false;
		}
		return rows == other.rows && cols == other.cols && Arrays.deepEquals(data, other.data);
	}

	@Override
	public int hashCode() {
		int result = Objects.hash(rows, cols);
		result = 31 * result + Arrays.deepHashCode(data);
		return result;
	}

	@Override
	public String toString() {
		return "Matrix{" +
				"rows=" + rows +
				", cols=" + cols +
				", data=" + Arrays.deepToString(data) +
				'}';
	}
}
