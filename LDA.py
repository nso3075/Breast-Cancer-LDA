import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Loading the dataset
data = pd.read_csv("data/diabetes_012_health_indicators_BRFSS2015.csv")

# Separate features and labels
X = data.iloc[:, 1:].values # features are all the columns minus first column
y = data.iloc[:, 0].values # features are the first column (diabetes type)

# Get class means, overall mean
classes = np.unique(y)
mean_vectors = {c: np.mean(X[y == c], axis=0) for c in classes}
mean_overall = np.mean(X, axis=0)

# normalizing the data: subtract the mean of each feature from each data point
for row in X:
    for col, _  in enumerate(row):
        row[col] -= mean_overall[col]

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

# getting eigenvals and eigenvectors of wc_ms^-1 * bc_ms
# Using pinv instead of inv so it will give pseudoinverse if real inverse isnt possible to calcuate
new_mat = np.linalg.pinv(wc_ms) @ bc_ms
eigenvalues, eigenvectors = np.linalg.eig(new_mat)

# sort the eigenvec and vals so the first one is most important
sorted_indices = np.argsort(-eigenvalues.real)
eigvals = eigenvalues[sorted_indices].real
eigvecs = eigenvectors[:, sorted_indices].real

# project the data onto top two eigen values --> 2d projection = good for visualization
W = eigvecs[:, :2]
X_lda = X @ W

blue_targets = X_lda[np.where(y == 0.0)]
red_targets = X_lda[np.where(y == 1.0)]
orange_targets = X_lda[np.where(y == 2.0)]

plt.scatter(
        blue_targets[:, 0],
        blue_targets[:, 1],
        marker="o",
        facecolors="none",
        edgecolors="blue",
    )

plt.scatter(
        red_targets[:, 0],
        red_targets[:, 1],
        marker="o",
        facecolors="none",
        edgecolors="red",
    )

plt.scatter(
        orange_targets[:, 0],
        orange_targets[:, 1],
        marker="o",
        facecolors="none",
        edgecolors="orange",
    )

plt.savefig("testing.png")