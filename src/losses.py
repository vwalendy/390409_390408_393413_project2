class MSE:
    @staticmethod
    def loss(y_true, y_pred):
        """
        Mean squared error, averaged over samples (and outputs).

        :param y_true: (array) target values, shape (N, C) or (N,)
        :param y_pred: (array) predicted values, same shape as y_true
        :return: (float)
        """
        return float(np.mean((y_pred - y_true) ** 2))

    @staticmethod
    def gradient(y_true, y_pred):
        """
        Gradient of MSE w.r.t. y_pred. Normalised by the number of samples
        so the scale matches the averaging used in .loss().
        """
        N = y_true.shape[0]
        return 2.0 * (y_pred - y_true) / N


class CrossEntropy:
    """
    Cross-entropy loss for classification, expects y_pred to be a
    probability distribution per row (e.g. output of Softmax) and
    y_true to be one-hot encoded.
    """

    @staticmethod
    def loss(y_true, y_pred):
        N = y_true.shape[0]
        eps = 1e-12
        return float(-np.sum(y_true * np.log(y_pred + eps)) / N)

    @staticmethod
    def gradient(y_true, y_pred):
        """
        Gradient of cross-entropy w.r.t. the pre-activation of the output
        layer, assuming Softmax output: (y_pred - y_true) / N.

        Because Softmax.gradient returns 1, multiplying this by the
        activation gradient in backprop leaves the correct delta.
        """
        N = y_true.shape[0]
        return (y_pred - y_true) / N