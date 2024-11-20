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
num_features = X.shape[1] # num features is equal to number of columns

# Represents variance of data points within each class relative to class mean
wc_ms = np.zeros((num_features, num_features)) # within class scatter matrix initialized with dimensions num_featuresXnum_features

# Measures the spread of the class means relative to the overall mean of the dataset.
bc_ms = np.zeros((num_features, num_features))

for c in classes:
    class_data = X[y==c] # getting data for each class
    mean_c = mean_vectors[c].reshape(-1, 1) # getting mean from precomputed vector and reshaping for multiplication so its a col vector
    
    # Getting sum of (X-uc)(X-uc)^T
    wc_ms += np.sum([(x.reshape(-1, 1) - mean_c) @ (x.reshape(-1, 1) - mean_c).T for x in class_data], axis=0)


# Between-class scatter matrix
for c in classes:
    num_samples_c = np.sum(y == c) # getting how many sampels there are in c
    mean_diff = (mean_vectors[c] - mean_overall).reshape(-1, 1)
    # getting sum of num_samples_c * (uc - u)(uc - u)^T by comparing mean sample for class and overall mean sample
    bc_ms += num_samples_c * (mean_diff @ mean_diff.T)
