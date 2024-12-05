import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

sd = 42
np.random.seed(sd)


# Function to perform LDA transformation
def perform_lda(X_train, y_train, X_test):
    classes = np.unique(y_train)
    mean_vectors = {c: np.mean(X_train[y_train == c], axis=0) for c in classes}
    mean_overall = np.mean(X_train, axis=0)

    X_train_norm = X_train - mean_overall
    X_test_norm = X_test - mean_overall

    num_features = X_train.shape[1]
    wc_ms = np.zeros((num_features, num_features))
    bc_ms = np.zeros((num_features, num_features))

    for c in classes:
        class_data = X_train[y_train == c]
        mean_c = mean_vectors[c].reshape(-1, 1)
        wc_ms += np.sum([(x.reshape(-1, 1) - mean_c) @ (x.reshape(-1, 1) - mean_c).T for x in class_data], axis=0)
        num_samples_c = class_data.shape[0]
        mean_diff = (mean_vectors[c] - mean_overall).reshape(-1, 1)
        bc_ms += num_samples_c * (mean_diff @ mean_diff.T)

    new_mat = np.linalg.pinv(wc_ms) @ bc_ms
    eigenvalues, eigenvectors = np.linalg.eig(new_mat)

    sorted_indices = np.argsort(-eigenvalues.real)
    eigvecs = eigenvectors[:, sorted_indices].real
    
    
    def project_data(n_components):
        W = eigvecs[:, :n_components]
        W = eigvecs[:, :n_components]
        X_train_lda = X_train_norm @ W
        X_test_lda = X_test_norm @ W
        return X_train_lda, X_test_lda
    
    return project_data

# Search function for LDA
def lda_search_fn(X_train_lda, y_train, X_test_lda, y_test):
    class_means_lda = {c: np.mean(X_train_lda[y_train == c], axis=0) for c in np.unique(y_train)}

    def predict(x):
        distances = {c: np.linalg.norm(x - mean) for c, mean in class_means_lda.items()}
        return min(distances, key=distances.get)

    y_pred = np.array([predict(x) for x in X_test_lda])
    return accuracy_score(y_test, y_pred)

# Search function for Random Forest
def rf_search_fn(X_train, y_train, X_test, y_test, n_estimators):
    rf_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=sd)
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)
    return accuracy_score(y_test, y_pred)

def plot_accuracies(x_values, accuracies, xlabel, ylabel, title, label=None):
    plt.plot(x_values, accuracies, marker='o', label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    if label:
        plt.legend()

def plot_accuracies_bar(rf_data):
    categories = list(rf_data.keys())
    values = list(rf_data.values())

    # Create the bar plot
    plt.figure(figsize=(8, 5))
    plt.bar(categories, values, color='skyblue', edgecolor='black')

    # Add labels and title
    plt.xlabel('Category')
    plt.ylabel('Accuracy')
    plt.title('Accuracy for Different Categories')
    plt.ylim(0.75, 0.85)  # Set minimum y-axis limit for better focus
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show plot
    plt.savefig("accuracies_rf_bar.png")


# Function to combine LDA and Random Forest in a pipeline
def lda_rf_pipeline(X_train, y_train, X_test, y_test, n_components_list, n_estimators_list):
    rf_data = {}
    classes = np.unique(y_train)
    max_components = min(len(classes) - 1, X_train.shape[1])
    
    compute_projection = perform_lda(X_train, y_train, X_test)

    lda_accuracies = []
    for n_components in n_components_list:
        print(f"Testing LDA with {n_components} components")
        X_train_lda, X_test_lda = compute_projection(n_components)
        acc = lda_search_fn(X_train_lda, y_train, X_test_lda, y_test)
        lda_accuracies.append(acc)
    plot_accuracies(n_components_list, lda_accuracies, "Number of Components", "Accuracy",
                    "LDA Accuracy by Number of Components", label="LDA")
    

    # RF Accuracy Plot with Overlays
    rf_no_lda_accuracies = []
    for n_estimators in n_estimators_list:
        rf_no_lda_acc = rf_search_fn(X_train, y_train, X_test, y_test, n_estimators)
        rf_no_lda_accuracies.append(rf_no_lda_acc)
        if n_estimators == max(n_estimators_list): rf_data["No LDA"] = rf_no_lda_acc
        rf_accuracies = []
        for n_components in n_components_list:
            print(f"Testing RFxLDA with {n_components} components and {n_estimators} estimators")
            X_train_lda, X_test_lda = compute_projection(n_components)
            acc = rf_search_fn(X_train_lda, y_train, X_test_lda, y_test, n_estimators)
            if n_estimators == max(n_estimators_list): rf_data[n_components] = acc
            rf_accuracies.append(acc)

        plot_accuracies(n_components_list, rf_accuracies, "Number of Components", "Accuracy",
                        f"RF Accuracy (n_estimators={n_estimators}) vs LDA", label=f"RF (n={n_estimators})")
        
    plt.legend()
    plt.savefig("lda_rf_accuracy_comparison.png")
    plt.show()


    for i, acc in enumerate(rf_no_lda_accuracies):
        print(f"RF with no LDA (n_estimators={n_estimators_list[i]}) accuracy: {acc}\n")
    
    plot_accuracies_bar(rf_data)


# Function to process data file and apply searches
def process_data_file(file_path, n_estimators_list):
    print(f"\nProcessing file: {file_path}")
    data = pd.read_csv(file_path)
    data = data.sample(frac=0.1, random_state=42) #DELETE LATER
    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values

    split_ratio = 0.8
    indices = np.random.permutation(len(X))
    split_index = int(len(X) * split_ratio)
    X_train, X_test = X[indices[:split_index]], X[indices[split_index:]]
    y_train, y_test = y[indices[:split_index]], y[indices[split_index:]]

    classes = np.unique(y_train)
    max_components = 11#min(len(classes) - 1, X_train.shape[1])
    n_components_list = list(range(1, max_components + 1))

    # print("\nLDA Grid Search:")
    # # grid_search_and_plot(X_train, y_train, X_test, y_test, n_components_list, "LDA", lda_search_fn)

    # print("\nRandom Forest Grid Search:")
    # # grid_search_and_plot(X_train, y_train, X_test, y_test, n_estimators_list, "Random Forest", rf_search_fn)

    print("\nLDA + Random Forest Pipeline:")
    lda_rf_pipeline(X_train, y_train, X_test, y_test, n_components_list, n_estimators_list)

# Define data files and parameters
data_files = [
    "data/diabetes_012_health_indicators_BRFSS2015.csv"
]
n_estimators_list = [25, 50, 100]

for file_path in data_files:
    process_data_file(file_path, n_estimators_list)


    