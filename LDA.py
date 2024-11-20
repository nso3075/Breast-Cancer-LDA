import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Loading the dataset
data = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv")

# Separate features and labels
X = data.iloc[:, 1:].values # features are all the columns minus first column
y = data.iloc[:, 0].values # features are the first column (diabetes type)

# Get class means, overall mean
classes = np.unique(y)
mean_vectors = {c: np.mean(X[y == c], axis=0) for c in classes}
mean_overall = np.mean(X, axis=0)

# Within-class scatter matrix