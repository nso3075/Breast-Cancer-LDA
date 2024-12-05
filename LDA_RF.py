import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Loading the dataset
data = pd.read_csv("data/diabetes_012_health_indicators_BRFSS2015.csv")


X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values 

# data split
split_ratio = 0.8
split_index = int(len(X) * split_ratio)
indices = np.random.permutation(len(X))

X_train, X_test = X[indices[:split_index]], X[indices[split_index:]]
y_train, y_test = y[indices[:split_index]], y[indices[split_index:]]

# Getinf class means and overall mean
classes = np.unique(y_train)
mean_vectors = {c: np.mean(X_train[y_train == c], axis=0) for c in classes}
mean_overall = np.mean(X_train, axis=0)

# Normalizing the data
X_train_norm = X_train - mean_overall
num_features = X_train.shape[1]
wc_ms = np.zeros((num_features, num_features))


wc_ms = np.zeros((num_features, num_features)) 

bc_ms = np.zeros((num_features, num_features))

for c in classes:
    class_data = X_train[y_train == c]
    mean_c = mean_vectors[c].reshape(-1, 1)

    # Within-class scatter matrix
    wc_ms += np.sum([(x.reshape(-1, 1) - mean_c) @ (x.reshape(-1, 1) - mean_c).T for x in class_data], axis=0)

    # Between-class scatter matrix
    num_samples_c = len(class_data)
    mean_diff = (mean_vectors[c] - mean_overall).reshape(-1, 1)
    bc_ms += num_samples_c * (mean_diff @ mean_diff.T)

# eigenvalues and eigenvectors of wc_ms^-1 * bc_ms
new_mat = np.linalg.pinv(wc_ms) @ bc_ms
eigenvalues, eigenvectors = np.linalg.eig(new_mat)

# Sorting the eigenvectors and eigenvalues
sorted_indices = np.argsort(-eigenvalues.real)
eigvals = eigenvalues[sorted_indices].real
eigvecs = eigenvectors[:, sorted_indices].real

# Project the data onto top two eigenvectors (for 2D visualization)
W2 = eigvecs[:, :2]
W10 = eigvecs[:, :10]

X_train_lda = X_train_norm @ W2
X_test_lda = (X_test - mean_overall) @ W2

# Visualize the 2D projection
blue_targets = X_train_lda[y_train == 0]
red_targets = X_train_lda[y_train == 1]

plt.scatter(blue_targets[:, 0], blue_targets[:, 1], marker="o", edgecolors="blue", facecolors="none", label="Class 0")
plt.scatter(red_targets[:, 0], red_targets[:, 1], marker="o", edgecolors="red", facecolors="none", label="Class 1")
plt.legend()
plt.title("LDA Projection (2D)")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.savefig("lda_projection_2D.png")
plt.close()

# Random Forest classifier with LDA projected data
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_lda, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test_lda)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")