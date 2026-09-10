import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, max_depth=100, min_sample_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y, depth=0)

    def _grow_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # stopping conditions
        if (
            depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_sample_split
        ):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # randomly select features
        feature_idxs = np.random.choice(n_features, self.n_features, replace=False)

        # find best split
        best_feature, best_threshold = self._best_split(X, y, feature_idxs)

        # split data
        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)

        # grow children
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return Node(best_feature, best_threshold, left, right)

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def _best_split(self, X, y, feature_idxs):
        best_gain = -1
        split_idx = None
        split_threshold = None

        for feat_idx in feature_idxs:
            X_col = X[:, feat_idx]
            thresholds = np.unique(X_col)

            for threshold in thresholds:
                gain = self._information_gain(y, X_col, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = threshold

        return split_idx, split_threshold

    def _information_gain(self, y, X_col, threshold):
        parent_entropy = self._entropy(y)

        left_idxs, right_idxs = self._split(X_col, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)

        e_l = self._entropy(y[left_idxs])
        e_r = self._entropy(y[right_idxs])

        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        return parent_entropy - child_entropy

    def _split(self, X_col, threshold):
        left_idxs = np.argwhere(X_col <= threshold).flatten()
        right_idxs = np.argwhere(X_col > threshold).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


if __name__ == "__main__":
    # Create dataset
    X, y = make_moons(
        n_samples=5000,
        noise=0.25,
        random_state=42
    )

    print(X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    tree = DecisionTree(
        max_depth=25,
        min_sample_split=20,
        n_features=2
    )

    tree.fit(X_train, y_train)

    y_train_pred = tree.predict(X_train)
    y_test_pred = tree.predict(X_test)

    print(f"Train Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"Test Accuracy : {accuracy_score(y_test, y_test_pred):.4f}")

def plot_decision_boundary(X, y, clf, step=0.02, cmap=plt.cm.coolwarm):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, step),
        np.arange(y_min, y_max, step)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors="k", s=15)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Random Forest Decision Boundary")
    plt.show()


# plot_decision_boundary(X,y,clf=tree)