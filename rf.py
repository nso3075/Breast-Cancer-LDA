import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# Just the RF for sanity reasons

# Load the dataset
data = pd.read_csv("data/diabetes_012_health_indicators_BRFSS2015.csv")

# Separate features and labels
X = data.iloc[:, 1:].values  # Features (excluding the first column)
y = data.iloc[:, 0].values   # Labels (first column)

# Split the data into training and testing sets
split_ratio = 0.8
split_index = int(len(X) * split_ratio)
indices = np.random.permutation(len(X))

X_train, X_test = X[indices[:split_index]], X[indices[split_index:]]
y_train, y_test = y[indices[:split_index]], y[indices[split_index:]]

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
