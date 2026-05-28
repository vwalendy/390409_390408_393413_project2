import argparse
import numpy as np

from src.methods.dummy_methods import DummyClassifier
from src.methods.mlp import MLP
from src.losses import MSE, CrossEntropy
from src.activations import Sigmoid, ReLU, Identity, Softmax
from src.methods.kmeans import KMeans
from src.utils import (
    normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn,
    label_to_onehot, onehot_to_label, get_n_classes,
)
import os

np.random.seed(100)


def main(args):
    """
    The main function of the script.

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end
                          of this file). Their value can be accessed as "args.argument".
    """


    dataset_path = args.data_path
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    ## 1. We first load the data.

    feature_data = np.load(dataset_path, allow_pickle=True)
    train_features, test_features, train_labels_reg, test_labels_reg, train_labels_classif, test_labels_classif = (
        feature_data['xtrain'],feature_data['xtest'],feature_data['ytrainreg'],
        feature_data['ytestreg'],feature_data['ytrainclassif'],feature_data['ytestclassif']
    )
    train_labels_classif = train_labels_classif.astype(int)
    test_labels_classif = test_labels_classif.astype(int)

    ## 2. Then we must prepare it. This is where you can create a validation set,
    #  normalize, add bias, etc.
    if args.task == "regression":
        train_labels = train_labels_reg
        test_labels = test_labels_reg
    else:
        train_labels = train_labels_classif
        test_labels = test_labels_classif

    # Make a validation set (it can overwrite xtest, ytest)
    if not args.test:
        N = train_features.shape[0]
        perm = np.random.permutation(N)
        n_val = int(0.2 * N)
        val_idx, tr_idx = perm[:n_val], perm[n_val:]

        test_features = train_features[val_idx]
        test_labels = train_labels[val_idx]
        train_features = train_features[tr_idx]
        train_labels = train_labels[tr_idx]

    means = train_features.mean(axis=0, keepdims=True)
    stds = train_features.std(axis=0, keepdims=True)
    stds[stds == 0] = 1.0
    train_features = normalize_fn(train_features, means, stds)
    test_features = normalize_fn(test_features, means, stds)


    ### WRITE YOUR CODE HERE to do any other data processing

    ## 3. Initialize the method you want to use.

    # Follow the "DummyClassifier" example for your methods
    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)

    elif args.method == "kmeans":
        method_obj = KMeans(K=args.K, max_iters=args.max_iters)

    elif args.method == "mlp":
        # Output size and head depend on the task.
        if args.task == "regression":
            out_dim = 1
            activations = (ReLU, Identity)
            loss = MSE
        else:
            out_dim = get_n_classes(train_labels)
            activations = (ReLU, Softmax)
            loss = CrossEntropy

        method_obj = MLP(
            dimensions=(train_features.shape[1], args.hidden_size, out_dim),
            activations=activations,
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")

    ## 4. Train and evaluate the method

    if args.task == "classification":

        if args.method == "mlp":
            C = get_n_classes(train_labels)
            y_train_oh = label_to_onehot(train_labels, C)
            method_obj.fit(train_features, y_train_oh, loss=loss,
                           epochs=args.max_iters, batch_size=args.batch_size,
                           learning_rate=args.lr)
            preds_train = onehot_to_label(method_obj.predict(train_features))
            preds = onehot_to_label(method_obj.predict(test_features))
        else:
            preds_train = method_obj.fit(train_features, train_labels)
            preds = method_obj.predict(test_features)

        acc = accuracy_fn(preds_train, train_labels)
        f1 = macrof1_fn(preds_train, train_labels)
        print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {f1:.6f}")

        acc = accuracy_fn(preds, test_labels)
        f1 = macrof1_fn(preds, test_labels)
        split = "Test" if args.test else "Validation"
        print(f"{split} set: accuracy = {acc:.3f}% - F1-score = {f1:.6f}")

    elif args.task == "regression":
        assert args.method != "kmeans", "You should use kmeans as a classification method"

        if args.method == "mlp":
            y_train = train_labels.reshape(-1, 1).astype(float)
            method_obj.fit(train_features, y_train, loss=loss,
                           epochs=args.max_iters, batch_size=args.batch_size,
                           learning_rate=args.lr)
            preds_train = method_obj.predict(train_features).ravel()
            preds = method_obj.predict(test_features).ravel()
        else:
            preds_train = method_obj.fit(train_features, train_labels)
            preds = method_obj.predict(test_features)

        loss_train = mse_fn(preds_train, train_labels)
        print(f"\nTrain set: MSE = {loss_train:.6f}")

        loss_test = mse_fn(preds, test_labels)
        split = "Test" if args.test else "Validation"
        print(f"{split} set: MSE = {loss_test:.6f}")

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="classification",
        type=str,
        help="classification / regression / clustering",
    )
    parser.add_argument(
        "--method",
        default="dummy_classifier",
        type=str,
        help="dummy_classifier / kmeans / mlp",
    )
    parser.add_argument(
        "--data_path",
        default="data/features.npz",
        type=str,
        help="path to your dataset CSV file",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=1,
        help="number of clusters datapoints used for kmeans",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="learning rate for methods with learning rate",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=100,
        help="max iters for methods which are iterative",
    )
    parser.add_argument(
        "--hidden_size", type=int, default=64,
        help="number of neurons in the MLP hidden layer",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="mini-batch size for MLP training",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="train on whole training data and evaluate on the test data, "
             "otherwise use a validation set",
    )
    # Feel free to add more arguments here if you need!

    args = parser.parse_args()
    main(args)
