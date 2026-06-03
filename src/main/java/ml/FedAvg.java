package ml;

import math.Matrix;

import java.util.List;
import java.util.Objects;

public final class FedAvg {
    private FedAvg() {
    }

    public static Matrix[] aggregate(List<NeuralNetwork> nodes) {
        Objects.requireNonNull(nodes, "nodes cannot be null");
        if (nodes.isEmpty()) {
            throw new IllegalArgumentException("nodes cannot be empty");
        }

        Matrix[] reference = nodes.get(0).getParameters();
        Matrix[] averaged = new Matrix[reference.length];

        for (int i = 0; i < reference.length; i++) {
            Matrix sum = reference[i];
            for (int n = 1; n < nodes.size(); n++) {
                Matrix[] params = nodes.get(n).getParameters();
                if (params.length != reference.length) {
                    throw new IllegalArgumentException("All networks must have the same number of parameters.");
                }
                if (params[i].getRows() != reference[i].getRows() || params[i].getCols() != reference[i].getCols()) {
                    throw new IllegalArgumentException("All networks must have matching parameter shapes.");
                }
                sum = sum.add(params[i]);
            }
            averaged[i] = sum.scale(1.0 / nodes.size());
        }

        return averaged;
    }
}

