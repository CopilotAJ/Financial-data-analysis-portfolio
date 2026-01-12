#The Manual implementation of SVM for Binary classification
#This is a basic SVM written using numpy only

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Firstly, load the dataset to be used
#Using the dataset with the indicators for AAPL (Apple)
df = pd.read_csv("stock_indicators_AAPL.csv")

#Create the binary target (if price went up the next day it will be 1, else will be 0)
df['price_up'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df.dropna(inplace=True)

#We are using only 2 features for visualization
X = df[['sma_5', 'ema_7']].values
y = df['price_up']

#Convert labels from (0,1) to (-1, 1) for SVM
#Mathematically, SVM works better with lables -1 and 1
y = np.where(y ==0, -1, 1)

#Normalize the features, so all featueres can be on the same scalee with mean 0 and std 1
X = (X -X.mean(axis=0)) / X.std(axis=0)


#SVM training usig Stochastic Gradient Descent (SGD)
#SGD is the most practical and popular method to implement SVM manaually.
class SimpleSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
    
    #
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.lr * y[idx]
                    
    def predict(self, X):
        approx = np.dot(X, self.w) + self.b
        return np.sign(approx).flatten()
    
    
#Multiple train/test splits: 60/40, 70/30, 80/20 and 90/10
#Store accuracy results
split_ratios = [0.6, 0.7, 0.8, 0.9]
accuracies = []

#Loop through the different train/test splits
for split in split_ratios:
    #round is used because int cancells decimals, just for the display.
    print(f"\n...Using the train/test split of: {round(split*100)}/{round((1-split)*100)} ...")
    split_idx = int(split * len(X))
    X_train, X_test =X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]


    #Initialize and train custiom SVM
    manual_svm = SimpleSVM()
    manual_svm.fit(X_train, y_train)

    #Make the predictions
    y_pred_manual = manual_svm.predict(X_test)

    #Convert the predictions back to (0,1)
    y_pred_binary = np.where(y_pred_manual == -1, 0, 1)
    y_test_binary = np.where(y_test == -1, 0, 1)

    #The accuracy calculation
    #comparing the predicted and actual values with 'np.mean' calculating the percentage of correct predictions
    accuracy = np.mean(y_pred_binary == y_test_binary)
    accuracies.append((f"{int(split*100)}/{int((1-split)*100)}", accuracy))
    print(f"\nManual SVM Accuracy on the test set: {accuracy:.2f}")
    
    #Save the last model for plotting the decision boundary
    if split == 0.9:
        final_model = manual_svm
        final_X_test = X_test
        final_y_test = y_test
    


#Save the accuracy results to csv
acc_df = pd.DataFrame(accuracies, columns=["The train/test split", "Accuracy"])
acc_df.to_csv("the manual svm_split accuracy.csv", index=False)
print("\nSaved the accuracy results to the manual svm_split accuracy.csv.")



#The plot for train/test split vs Accuracy
splits = [r[0] for r in accuracies]#extracts the train percentage
values = [r[1] for r in accuracies]#extracts the accracy for each split.

plt.figure(figsize=(8, 6))
plt.plot(splits, values, marker='o', linestyle='-', color='orange')
plt.title("The Manual SVM model Accuracy across the train/test splits")
plt.xlabel("Train/Test Split")
plt.ylabel("Accuracy")
plt.grid(True)
plt.tight_layout()
plt.savefig("the_manual_svm_acuuracy_comparison.png")
print("The accuracy comparison plot has been saved as 'the_manual_svm_acuuracy_comparison.png'")
plt.show()


#Visualizing Decision boundary for the 90/10 split (2D)
#A helper function to visually show the SVM decision boundary
#Decision boundary is the line that separates different classes.
def plot_decision_boundary(X, y, model):
    def decision_function(x):
        return -(model.w[0] * x + model.b) / model.w[1]
    
    plt.figure(figsize=(8, 6))
    for idx, label in enumerate(np.unique(y)):
        plt.scatter(X[y == label, 0], X[y == label, 1], label=f"Class {label}", edgecolors='k')
        
    x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    plt.plot(x_vals, decision_function(x_vals), '--', color='red', label='Decision Boundary')  
    plt.xlabel("sma_5 (normalized)")
    plt.ylabel("ema_7 (normalized)")
    plt.title("The Maunal SVM Decision Boundary for 90/10 split")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("manual_svm_decisionboundary.png")
    print("\nSaved the 'manual_svm_decisionboundary' which shows the decision boundary for 90/10 split.")
    plt.show()

plot_decision_boundary(final_X_test, final_y_test, final_model)


        
        
