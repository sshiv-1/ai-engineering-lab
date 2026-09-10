import numpy as np
import matplotlib.pyplot as plt


class WhenTheNightCrawls:
    def __init__(self,lr=0.01,epochs=1000):
        self.lr=lr
        self.epochs=epochs
        self.w=None
        self.b=None



    def fit(self,X,y):
        self.losses=[]
        n_samples,n_features=X.shape

        self.w=np.zeros(n_features)
        self.b=0.0

        for epoch in range(self.epochs):
            y_pred=np.dot(X,self.w) + self.b
            loss=np.mean(y_pred-y)**2
            self.losses.append(loss)
            
            dw = (2 / n_samples) * np.dot(X.T, (y_pred - y))
            db=(2/n_samples)*(np.sum(y_pred-y))

            self.w-=self.lr*dw
            self.b-=self.lr*db


    def predict(self,X):
        return np.dot(X, self.w) + self.b
    
    def r2_score(self, y, y_pred):
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot
    
    def plot_predictions(self,X,y,y_pred):
        plt.scatter(X, y, color="blue", label="Actual")
        plt.plot(X, y_pred, color="red", label="Predicted")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.show()


    def plot_loss(self,losses):
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Training Loss Curve")
        plt.show()



X = np.array([[1], [2], [3], [4], [5]])
y = np.array([3, 5, 7, 9, 11]) 
model = WhenTheNightCrawls(lr=0.01, epochs=1000)
model.fit(X, y)
y_pred=model.predict(X)
print(model.w)  
print(model.b)  
print(y_pred)
print(f"Accuracy:{model.r2_score(y,y_pred)}")
# model.plot_predictions(X,y,y_pred)
model.plot_loss(model.losses)