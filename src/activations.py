import numpy as np

class Sigmoid:
    @staticmethod
    def forward(z):
        out = np.empty_like(z, dtype=float)
        pos = z >= 0
        neg = ~pos
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[neg])
        out[neg] = ez / (1.0 + ez)
        return out

    @staticmethod
    def gradient(z):
        s = Sigmoid.forward(z)
        return s * (1.0 - s)

class ReLU:
    @staticmethod
    def forward(z):
        return np.maximum(0.0, z)

    @staticmethod
    def gradient(z):
        # Derivative is 1 where z > 0, else 0. Convention: 0 at z == 0.
        return (z > 0).astype(float)

class ReLU:
    @staticmethod
    def forward(z):
        return np.maximum(0.0, z)

    @staticmethod
    def gradient(z):
        # Derivative is 1 where z > 0, else 0. Convention: 0 at z == 0.
        return (z > 0).astype(float)


class Identity:
    """Linear activation. Used on the output layer for regression."""

    @staticmethod
    def forward(z):
        return z

    @staticmethod
    def gradient(z):
        return np.ones_like(z, dtype=float)


class Softmax:
    """
    Softmax activation for the output layer in classification.

    Note: gradient() returns ones because we pair Softmax with the
    CrossEntropy loss. The combined gradient (y_pred - y_true) is handled
    directly by CrossEntropy.gradient, so the activation contributes a
    factor of 1 in the backprop chain for the output layer.
    """

    @staticmethod
    def forward(z):
        z_shift = z - np.max(z, axis=1, keepdims=True)
        ez = np.exp(z_shift)
        return ez / np.sum(ez, axis=1, keepdims=True)

    @staticmethod
    def gradient(z):
        return np.ones_like(z, dtype=float)