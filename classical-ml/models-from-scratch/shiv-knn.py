import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class KNNClf:
    
    def __init__(self,*,k):
        self.k = k

    def fit(self,X,y):
        self.X_train=X
        self.y_train=y

    def eucledian_distance(self,query):
        return np.sum((self.X_train - query)**2,axis=1)
    
    def predict_one(self,x_q):
        distances=self.eucledian_distance(x_q)
        nearest_index=np.argpartition(distances,self.k)[:self.k] #sort
        knn_distances=distances[nearest_index]
        labels=self.y_train[nearest_index]  #extracted the class a or b

        theta=1e-8  #avoid zero division
        weights=1/(knn_distances+theta)

        vote=np.bincount(labels,weights=weights)  #if belongs to respective class bin[i]=1+weight
        return np.argmax(vote)

    
    def predict(self,X_test):
        predictions=[]
        for x_q in X_test:
            y_pred=self.predict_one(x_q)
            predictions.append(y_pred)
        return np.array(predictions)
        

    def accuracy(self,y_true, y_pred):
        return np.mean(y_true == y_pred)



np.random.seed(42)

# Class 0
X0 = np.random.randn(100, 2) + np.array([0, 0])
y0 = np.zeros(100, dtype=int)

# Class 1
X1 = np.random.randn(100, 2) + np.array([3, 3])
y1 = np.ones(100, dtype=int)

X = np.vstack((X0, X1))
y = np.hstack((y0, y1))


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,      
    random_state=42,     
    stratify=y           
)

knn = KNNClf(k=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

acc = knn.accuracy(y_test, y_pred)

print("Accuracy:", acc)

plt.scatter(
    X_test[:, 0],
    X_test[:, 1],
    c=y_pred,
    cmap="coolwarm",
    edgecolor="k"
)
plt.title("KNN Predictions (Test Set)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()





