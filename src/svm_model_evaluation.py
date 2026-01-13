#Import the necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


def process_stock(ticker):
    print(f"\n...Processing {ticker}...")


 #Load dataset with the technical indicators
    filepath = f"stock_indicators_{ticker}.csv"
    df = pd.read_csv(filepath)


    #The Target (y) variable is 'price_up'
    #The task is going to be a binary classification problem
    #We are comparing the next day's closing price to today
    #If tomorrow's price goes up it is '1', otherwise 0
    df['price_up'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    #Drop last row as there's no next day and will return NaN
    df.dropna(inplace=True)


    #The features (inputs(x)) to be used in the model
    features = ['sma_5', 'ema_7', 'rsi_9', 'macd_line', 'macd_signal']
    X = df[features]
    y = df['price_up']

    #Splitting dataset into training and testing sets (80% train, and 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    #Normalizing the features (X) by using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #Initializing 4 machine learning models
    models = {
        "Support Vector Machine (SVM)": SVC(kernel='rbf', C=1.0),
        "Logistic Regression Model": LogisticRegression(),
        "Random Forest Classifier Model": RandomForestClassifier(n_estimators=100, random_state=42),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5)
    }


    #To create a comparison table of model performance
    #Where to store the predictions of each model
    predictions = {}
    #Where to store the results
    results = {
        "Model": [],
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1-Score": []
    }

    #To go through all models to train, predict and evaluate
    for name, model in models.items():
        #Train thr model on the training data
        model.fit(X_train_scaled, y_train)
    
        #Predict the test data
        y_pred = model.predict(X_test_scaled)
        predictions[name] = y_pred

        #Function to compute and store the metrics
        results["Model"].append(name)
        results["Accuracy"].append(accuracy_score(y_test, y_pred))
        results["Precision"].append(precision_score(y_test, y_pred, zero_division=0))
        results["Recall"].append(recall_score(y_test, y_pred, zero_division=0))
        results['F1-Score'].append(f1_score(y_test, y_pred, zero_division=0))
    
         #Print the detailed performance report for each of the machine learinig model
        print(f"{name} Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
    
         #Save the models predictions for comparison
    df_test = df.loc[X_test.index].copy()
    for model_name, y_pred in predictions.items():
        df_test[f"{model_name}_prediction"] = y_pred
    df_test.to_csv(f"model_predictions_{ticker}.csv", index=False)
    print(f"The predictions are saved to model_predictions_{ticker}.csv")
    
    
    #Save the model evaluation metrics to a csv file
    comparison_df = pd.DataFrame(results)
    comparison_df.to_csv(f"model_comparison_{ticker}.csv", index=False)
    print(f"The model comparison to model_comparison_{ticker}.csv")

    #The Visual Comparison of Actual vs All models predicted price movements
    #Using the test part for plotting
    plt.figure(figsize=(14, 7))
    x_axis = df_test['Date'] if 'Date' in df_test.columns else df_test.index

    #'marker=o' to visually highlight individual prediction
    #For Actual values
    plt.plot(x_axis, y_test.values, label='Actual Price Movement', marker='o', linestyle='--', color='black')

    #The predictions for the models
    #model.keys() returns the list of all the models used and then zip pairs each of the odel name with the color
    for model_name, color in zip(models.keys(), ['red', 'blue', 'green', 'purple']):
        plt.plot(x_axis,  df_test[f"{model_name}_prediction"], label=f"{model_name} Prediction",
             marker='x', linestyle='-')

    #The plot settings
    plt.title(f"Actual vs Predicted Price Movement for {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Price Movement (1 = Up, 0 = Down)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("actual_allmodels_prediction_{ticker}.png")
    print("\nSaved the comparison visualization plot as 'actual_allmodels_prediction_{ticker}.png'.")
    plt.show()

#This is to call the function you want to analyze
#process_stock("AAPL")
process_stock("NFLX")
