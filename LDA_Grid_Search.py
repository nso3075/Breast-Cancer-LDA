import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Random seed so results are the same, splits are done the same every time
# np.random.seed(42)

# Function to perform LDA and evaluate performance
def lda_grid_search(X_train, y_train, X_test, y_test, n_components_list):
    accuracies = []
    classes = np.unique(y_train)
    num_classes = len(classes)
    max_components = min(num_classes - 1, X_train.shape[1])

    for n_components in n_components_list:
        # Ensure n_components does not exceed max_components
        n_components = min(n_components, max_components)

        # Compute class means and overall mean
        mean_vectors = {c: np.mean(X_train[y_train == c], axis=0) for c in classes}
        mean_overall = np.mean(X_train, axis=0)

        # Normalize the data
        X_train_norm = X_train - mean_overall
        X_test_norm = X_test - mean_overall
        num_features = X_train.shape[1]

        # Initialize scatter matrices
        wc_ms = np.zeros((num_features, num_features))  # Within-class scatter matrix
        bc_ms = np.zeros((num_features, num_features))  # Between-class scatter matrix

        for c in classes:
            class_data = X_train[y_train == c]
            mean_c = mean_vectors[c].reshape(-1, 1)
            # Within-class scatter matrix
            wc_ms += np.sum([(x.reshape(-1, 1) - mean_c) @ (x.reshape(-1, 1) - mean_c).T for x in class_data], axis=0)
            # Between-class scatter matrix
            num_samples_c = class_data.shape[0]
            mean_diff = (mean_vectors[c] - mean_overall).reshape(-1, 1)
            bc_ms += num_samples_c * (mean_diff @ mean_diff.T)

        # Solve the eigenvalue problem for the matrix wc_ms^-1 * bc_ms
        new_mat = np.linalg.pinv(wc_ms) @ bc_ms
        eigenvalues, eigenvectors = np.linalg.eig(new_mat)

        # Sort the eigenvectors by decreasing eigenvalues
        sorted_indices = np.argsort(-eigenvalues.real)
        eigvals = eigenvalues[sorted_indices].real
        eigvecs = eigenvectors[:, sorted_indices].real

        # Select the top 'n_components' eigenvectors
        W = eigvecs[:, :n_components]

        # Project the data onto the new subspace
        X_train_lda = X_train_norm @ W
        X_test_lda = X_test_norm @ W

        # LDA Classifier: Nearest class mean
        class_means_lda = {c: np.mean(X_train_lda[y_train == c], axis=0) for c in classes}

        def predict(x):
            distances = {c: np.linalg.norm(x - mean) for c, mean in class_means_lda.items()}
            return min(distances, key=distances.get)

        # Predict for test data
        y_pred = np.array([predict(x) for x in X_test_lda])
        accuracy = np.mean(y_pred == y_test)
        accuracies.append(accuracy)

        print(f"n_components: {n_components}, Accuracy: {accuracy * 100:.2f}%")

    # Plotting accuracy vs n_components
    plt.figure(figsize=(10, 6))
    plt.plot(n_components_list, [a * 100 for a in accuracies], marker='o')
    plt.title('LDA Classifier Accuracy vs Number of Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)

    # Save plot to a file
    plt.savefig('lda_accuracy_vs_n_components.png')
    plt.close()

    return accuracies

# Function for Random Forest grid search
def rf_grid_search(X_train, y_train, X_test, y_test, n_estimators_list):
    accuracies = []
    for n_estimators in n_estimators_list:
        # Random Forest classifier with specified number of estimators
        rf_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        rf_classifier.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = rf_classifier.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

        print(f"n_estimators: {n_estimators}, Accuracy: {accuracy * 100:.2f}%")

    # Plotting accuracy vs n_estimators
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_list, [a * 100 for a in accuracies], marker='o')
    plt.title('Random Forest Accuracy vs Number of Estimators')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)

    plt.savefig('rf_accuracy_vs_n_estimators.png')
    plt.close()

    return accuracies

# Function to perform LDA + Random Forest pipeline
def lda_rf_pipeline(X_train, y_train, X_test, y_test, n_components_list, n_estimators_list):
    classes = np.unique(y_train)
    num_classes = len(classes)
    max_components = min(num_classes - 1, X_train.shape[1])

    for n_components in n_components_list:
        # Ensure n_components does not exceed max_components
        n_components = min(n_components, max_components)

        # Compute class means and overall mean
        mean_vectors = {c: np.mean(X_train[y_train == c], axis=0) for c in classes}
        mean_overall = np.mean(X_train, axis=0)

        # Normalize the data
        X_train_norm = X_train - mean_overall
        X_test_norm = X_test - mean_overall
        num_features = X_train.shape[1]

        # Initialize scatter matrices
        wc_ms = np.zeros((num_features, num_features))  # Within-class scatter matrix
        bc_ms = np.zeros((num_features, num_features))  # Between-class scatter matrix

        for c in classes:
            class_data = X_train[y_train == c]
            mean_c = mean_vectors[c].reshape(-1, 1)
            # Within-class scatter matrix
            wc_ms += np.sum([(x.reshape(-1, 1) - mean_c) @ (x.reshape(-1, 1) - mean_c).T for x in class_data], axis=0)
            # Between-class scatter matrix
            num_samples_c = class_data.shape[0]
            mean_diff = (mean_vectors[c] - mean_overall).reshape(-1, 1)
            bc_ms += num_samples_c * (mean_diff @ mean_diff.T)

        # Solve the eigenvalue problem for the matrix wc_ms^-1 * bc_ms
        new_mat = np.linalg.pinv(wc_ms) @ bc_ms
        eigenvalues, eigenvectors = np.linalg.eig(new_mat)

        # Sort the eigenvectors by decreasing eigenvalues
        sorted_indices = np.argsort(-eigenvalues.real)
        eigvals = eigenvalues[sorted_indices].real
        eigvecs = eigenvectors[:, sorted_indices].real

        # Select the top 'n_components' eigenvectors
        W = eigvecs[:, :n_components]

        # Project the data onto the new subspace
        X_train_lda = X_train_norm @ W
        X_test_lda = X_test_norm @ W

        print(f"\nRandom Forest Grid Search with n_components = {n_components}")

        accuracies = []
        for n_estimators in n_estimators_list:
            # Random Forest classifier with specified number of estimators
            rf_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            rf_classifier.fit(X_train_lda, y_train)

            # Make predictions on the test set
            y_pred = rf_classifier.predict(X_test_lda)

            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)

            print(f"n_estimators: {n_estimators}, Accuracy: {accuracy * 100:.2f}%")

        # Plotting accuracy vs n_estimators for current n_components
        plt.figure(figsize=(10, 6))
        plt.plot(n_estimators_list, [a * 100 for a in accuracies], marker='o')
        plt.title(f'Random Forest Accuracy vs Number of Estimators (n_components={n_components})')
        plt.xlabel('Number of Estimators')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)

        plt.savefig(f'rf_accuracy_vs_n_estimators_n_components_{n_components}.png')
        plt.close()

# Function to process each data file
def process_data_file(file_path, n_estimators_list):
    print(f"\nProcessing file: {file_path}")

    # Load data
    data = pd.read_csv(file_path)

    # Adjust feature and label extraction based on the dataset structure
    X = data.iloc[:, 1:].values  # Features
    y = data.iloc[:, 0].values   # Labels

    # Split data
    split_ratio = 0.8
    indices = np.random.permutation(len(X))
    split_index = int(len(X) * split_ratio)
    X_train, X_test = X[indices[:split_index]], X[indices[split_index:]]
    y_train, y_test = y[indices[:split_index]], y[indices[split_index:]]

    # Determine the maximum number of components
    classes = np.unique(y_train)
    num_classes = len(classes)
    max_components = min(num_classes - 1, X_train.shape[1])

    # Define the range of n_components to test
    n_components_list = list(range(1, max_components + 1))

    # Perform LDA grid search
    print("\nPerforming LDA Grid Search:")
    lda_accuracies = lda_grid_search(X_train, y_train, X_test, y_test, n_components_list)

    # Perform Random Forest grid search without LDA
    print("\nPerforming Random Forest Grid Search without LDA:")
    rf_accuracies = rf_grid_search(X_train, y_train, X_test, y_test, n_estimators_list)

    # Perform LDA + Random Forest pipeline
    print("\nPerforming LDA + Random Forest Grid Search:")
    lda_rf_pipeline(X_train, y_train, X_test, y_test, n_components_list, n_estimators_list)

# List of data files
data_files = [
    "data/diabetes_012_health_indicators_BRFSS2015.csv",
    # Can add the other data files here, it would be a pain to wait for it all to process though
    # "data/another_dataset.csv",
]

# Define the range of n_estimators to test for Random Forest
n_estimators_list = [50, 100, 150, 200]

# Process each data file
for file_path in data_files:
    process_data_file(file_path, n_estimators_list)
