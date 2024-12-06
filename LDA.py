import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


# Loading the dataset
data = pd.read_csv("data/diabetes_012_health_indicators_BRFSS2015.csv")


# Separate features and labels
X = data.iloc[:, 1:].values # features are all the columns minus first column
y = data.iloc[:, 0].values # features are the first column (diabetes type)


# Spit data
split_ratio = 0.8
split_index = int(len(X) * split_ratio)
indices = np.random.permutation(len(X))

X_train, X_test = X[indices[:split_index]], X[indices[split_index:]]
y_train, y_test = y[indices[:split_index]], y[indices[split_index:]]

# Get class means, overall mean
classes = np.unique(y_train)
mean_vectors = {c: np.mean(X_train[y_train == c], axis=0) for c in classes}
mean_overall = np.mean(X_train, axis=0)


# normalizing the data: subtract the mean of each feature from each data point
X_train_norm = X_train - mean_overall
num_features = X_train.shape[1]
wc_ms = np.zeros((num_features, num_features))

# Within-class scatter matrix
num_features = X.shape[1] # num features is equal to number of columns

# Represents variance of data points within each class relative to class mean
wc_ms = np.zeros((num_features, num_features)) # within class scatter matrix initialized with dimensions num_featuresXnum_features

# Measures the spread of the class means relative to the overall mean of the dataset.
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

# Getting eigenvals and eigenvectors of wc_ms^-1 * bc_ms
# Using pinv instead of inv so it will give pseudoinverse if real inverse isnt possible to calcuate
new_mat = np.linalg.pinv(wc_ms) @ bc_ms
eigenvalues, eigenvectors = np.linalg.eig(new_mat)

# sort the eigenvec and vals so the first one is most important
sorted_indices = np.argsort(-eigenvalues.real)
eigvals = eigenvalues[sorted_indices].real
eigvecs = eigenvectors[:, sorted_indices].real

# project the data onto top two eigen values --> 2d projection = good for visualization
W2 = eigvecs[:, :2]
W10 = eigvecs[:, :10]
X_train_lda = X_train_norm @ W2
X_test_lda = (X_test - mean_overall) @ W2

blue_targets = X_train_lda[y_train == 0.0]
red_targets = X_train_lda[y_train == 1.0]
orange_targets = X_train_lda[y_train == 2.0]

plt.scatter(blue_targets[:, 0], blue_targets[:, 1], marker="o", edgecolors="blue", facecolors="none", label="Class 0")
plt.scatter(red_targets[:, 0], red_targets[:, 1], marker="o", edgecolors="red", facecolors="none", label="Class 1")
plt.scatter(orange_targets[:, 0], orange_targets[:, 1], marker="o", edgecolors="orange", facecolors="none", label="Class 2")

plt.legend()
plt.title("LDA Projection (2D)")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.savefig("lda_projection_2D.png")
plt.close()

#LDA classifier
class_means_lda = {c: np.mean(X_train_lda[y_train == c], axis=0) for c in classes}


def predict(x, class_means):
    distances = {c: np.linalg.norm(x - mean) for c, mean in class_means.items()}
    return min(distances, key=distances.get)

# Predict for test data
y_pred = np.array([predict(x, class_means_lda) for x in X_test_lda])
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
# try in 3d space, try in 1d space
# make classifier and asses in different spaces

# Confusion matrix function
def conf_matrix(y_pred, y_true, title=None, print_results=True):
    confusions = np.zeros((3,3), dtype=int)
    for pred, true in zip(y_pred, y_true):
        confusions[int(pred)][int(true)] += 1

    # compute the recognition rate
    inputs_correct = np.sum([confusions[i][i] for i in range(3)])
    inputs_total = np.sum(confusions)
    recognition_rate = inputs_correct / inputs_total * 100

    # Print results
    if print_results:
        if title:
            print("\n>>> " + title)
        print(
            f"\n    Recognition rate (correct / inputs):\n  {recognition_rate:.2f}%\n"
        )
        print("\tConfusion Matrix:")
        print("\t\t| 0: No Diabetes | 1: Type One | 2: Type Two")
        print("     ---------------------------------------------")
        for i, row in enumerate(confusions):
            print(
                f"{i}: Pred-Class-{i} | "
                + " ".join(f"{val:10d}" for val in row)
            )
    print("\n\n")
    return confusions, recognition_rate
# Create Confusion matrix and print it
conf_matrix(y_pred, y_test, title = "LDA Classifier Results")
# Save the projected data and the labels into npy files
np.save("lda_results/X_train_lda.npy", X_train_lda)
np.save("lda_results/y_train.npy", y_train)
np.save("lda_results/X_test_lda.npy", X_test_lda)
np.save("lda_results/y_test.npy", y_test)
print("LDA Results Saved.")