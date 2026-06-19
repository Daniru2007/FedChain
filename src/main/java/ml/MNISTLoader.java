package ml;

import math.Matrix;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.zip.GZIPInputStream;

/**
 * Reads the MNIST dataset from raw IDX binary files (optionally gzip-compressed).
 *
 * <p>Expected file layout:
 * <ul>
 *   <li>Images – magic 0x00000803, count, rows, cols, then pixel bytes (0–255)</li>
 *   <li>Labels – magic 0x00000801, count, then label bytes (0–9)</li>
 * </ul>
 *
 * <p>Each image is returned as a {@code (rows*cols) × 1} column-vector Matrix with
 * pixel values normalised to [0, 1] by dividing by 255.
 * Each label is returned as a {@code 10 × 1} one-hot column-vector Matrix.
 */
public final class MNISTLoader {

    private static final int IMAGE_MAGIC = 0x00000803;
    private static final int LABEL_MAGIC = 0x00000801;

    private MNISTLoader() {}

    /**
     * Loads MNIST images from an IDX file (plain or gzip).
     *
     * @param path path to the IDX images file (e.g. {@code train-images-idx3-ubyte})
     * @return array of {@code (rows*cols) × 1} normalised column-vector Matrices
     * @throws IOException              on any I/O error
     * @throws IllegalArgumentException if the file magic number is wrong
     */
    public static Matrix[] loadImages(String path) throws IOException {
        try (DataInputStream in = openStream(path)) {
            int magic = in.readInt();
            if (magic != IMAGE_MAGIC) {
                throw new IllegalArgumentException(
                        "Invalid MNIST image file magic: expected 0x" +
                        Integer.toHexString(IMAGE_MAGIC) + " but got 0x" +
                        Integer.toHexString(magic));
            }
            int count  = in.readInt();
            int rows   = in.readInt();
            int cols   = in.readInt();
            int pixels = rows * cols;

            Matrix[] images = new Matrix[count];
            byte[] buffer   = new byte[pixels];

            for (int n = 0; n < count; n++) {
                int bytesRead = 0;
                while (bytesRead < pixels) {
                    int r = in.read(buffer, bytesRead, pixels - bytesRead);
                    if (r == -1) throw new IOException("Unexpected end of image file at image " + n);
                    bytesRead += r;
                }
                double[][] col = new double[pixels][1];
                for (int i = 0; i < pixels; i++) {
                    // byte is signed in Java; mask to get unsigned value in [0, 255]
                    col[i][0] = (buffer[i] & 0xFF) / 255.0;
                }
                images[n] = new Matrix(col);
            }
            return images;
        }
    }

    /**
     * Loads MNIST labels from an IDX file (plain or gzip).
     *
     * @param path path to the IDX labels file (e.g. {@code train-labels-idx1-ubyte})
     * @return array of {@code 10 × 1} one-hot column-vector Matrices
     * @throws IOException              on any I/O error
     * @throws IllegalArgumentException if the file magic number is wrong
     */
    public static Matrix[] loadLabels(String path) throws IOException {
        try (DataInputStream in = openStream(path)) {
            int magic = in.readInt();
            if (magic != LABEL_MAGIC) {
                throw new IllegalArgumentException(
                        "Invalid MNIST label file magic: expected 0x" +
                        Integer.toHexString(LABEL_MAGIC) + " but got 0x" +
                        Integer.toHexString(magic));
            }
            int count = in.readInt();

            Matrix[] labels = new Matrix[count];
            for (int n = 0; n < count; n++) {
                int digit = in.readUnsignedByte();
                double[][] oneHot = new double[10][1];
                oneHot[digit][0] = 1.0;
                labels[n] = new Matrix(oneHot);
            }
            return labels;
        }
    }

    /**
     * Opens a DataInputStream for {@code path}, transparently decompressing
     * gzip files (detected by a {@code .gz} suffix, case-insensitive).
     */
    private static DataInputStream openStream(String path) throws IOException {
        FileInputStream fis = new FileInputStream(path);
        if (path.toLowerCase().endsWith(".gz")) {
            return new DataInputStream(new GZIPInputStream(fis));
        }
        return new DataInputStream(fis);
    }
}
