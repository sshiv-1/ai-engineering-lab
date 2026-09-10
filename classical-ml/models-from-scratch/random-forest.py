import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from DecisionTree import DecisionTree
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class RandomForest:
    def __init__(self,n_trees=10,max_depth=25,min_sample_split=2,n_features=None):
        self.n_trees=n_trees
        self.max_depth=max_depth
        self.min_sample_split=min_sample_split
        self.n_features=n_features
        self.trees=[]

    def fit(self,X,y):
        self.trees=[]
        #create n decision trees
        for _ in range(self.n_trees):
            X_sample,y_sample=self.bootstrap_samples(X,y)
            tree=DecisionTree(max_depth=self.max_depth,min_sample_split=self.min_sample_split,n_features=self.n_features)
            tree.fit(X_sample,y_sample)
            self.trees.append(tree)

    def bootstrap_samples(self,X,y):
        n_samples=X.shape[0]
        idxs=np.random.choice(n_samples,size=n_samples,replace=True)
        return X[idxs],y[idxs]
    
    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]



    def predict(self,X):
        pred=np.array([tree.predict(X) for tree in self.trees])
        #fetch predictions from each tree for this point p
        tree_pred=np.swapaxes(pred,0,1)
        final_pred=np.array([self._most_common_label(prediction) for prediction in tree_pred])

        return final_pred
    

    # Create dataset
X, y = make_moons(
    n_samples=7000,
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


forest = RandomForest(
    n_trees=10,
    max_depth=25,
    min_sample_split=20,
    n_features=2
)

forest.fit(X_train, y_train)

# Predictions
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

# Accuracy
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy : {test_acc:.4f}")


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


plot_decision_boundary(X,y,clf=forest)


