import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def make_classification(n_samples=5000,flip_prob=0.15):
    np.random.seed(42)

    # Overlapping Gaussians
    X0 = np.random.randn(2, n_samples // 2) + np.array([[-1.0], [-1.0]])
    X1 = np.random.randn(2, n_samples // 2) + np.array([[1.0], [1.0]])

    X = np.hstack((X0, X1))
    y = np.hstack((np.zeros(n_samples // 2), np.ones(n_samples // 2)))

    # Flip labels (irreducible noise)
    flip_mask = np.random.rand(n_samples) < flip_prob
    y[flip_mask] = 1 - y[flip_mask]

    # Shuffle
    idx = np.random.permutation(n_samples)
    X = X[:, idx]
    y = y[idx]

    return X, y

def smooth(losses, window=50):
    return np.convolve(losses, np.ones(window)/window, mode='valid')


class LogisticRegression:
    def __init__(self,lr=0.01,epochs=1000):
        self.w=None
        self.b=None
        self.lr=lr
        self.epochs=epochs

    def fit(self,X,y):
        self.losses=[]
        n_features,n_samples=X.shape

        self.w=np.zeros(n_features)
        self.b=0.0

        for _ in range(self.epochs):
            z=np.dot(self.w,X)+self.b
            y_pred=sigmoid(z)

            loss = -np.mean(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))
            self.losses.append(loss)


            dw = (1/n_samples) * np.dot(X, (y_pred - y).T)
            db = (1/n_samples) * np.sum(y_pred - y)

            self.w-=self.lr*dw
            self.b-=self.lr*db  

    def fit_sgd(self, X, y, batch_size=64):
        self.losses = []
        n_features, n_samples = X.shape

        self.w = np.zeros(n_features)
        self.b = 0.0

        for epoch in range(self.epochs):

            # 1️⃣ Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[:, indices]
            y_shuffled = y[indices]

            epoch_loss = 0.0

            # 2️⃣ Mini-batches
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[:, i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                m = X_batch.shape[1]

                # Forward pass
                z = np.dot(self.w, X_batch) + self.b
                y_pred = sigmoid(z)

                # Loss (batch)
                loss = -np.mean(
                    y_batch * np.log(y_pred) +
                    (1 - y_batch) * np.log(1 - y_pred)
                )
                epoch_loss += loss

                # Gradients
                dz = y_pred - y_batch
                dw = (1/m) * np.dot(X_batch, dz.T)
                db = (1/m) * np.sum(dz)

                # Parameter update
                self.w -= self.lr * dw
                self.b -= self.lr * db

            # Average loss per epoch
            self.losses.append(epoch_loss / (n_samples // batch_size))
        

    def predict(self, X, threshold=0.5):
        z = np.dot(self.w, X) + self.b
        y_pred = sigmoid(z)
        return (y_pred >= threshold).astype(int)
    
    def accuracy(self, X, y):
        y_hat = self.predict(X)
        return np.mean(y_hat == y)
    
    def plot_loss(self,losses):
        plt.plot(smooth(losses))
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Training Loss Curve")
        plt.show()

X, y = make_classification(n_samples=20000)

model = LogisticRegression(lr=0.1, epochs=1000)
model.fit_sgd(X, y,batch_size=64)

print("Predictions:", model.predict(X))
print("Accuracy:", model.accuracy(X, y))
model.plot_loss(model.losses)


