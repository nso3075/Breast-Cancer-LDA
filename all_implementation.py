import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

sd = 42
np.random.seed(sd)
colors = ['#008080', '#0373d3', '#816cce', '#00ebeb', '#9fa6e9', '#2f1c91']

def graph_2D_scatter(X_train, y_train, eigvecs):

    plt.figure(facecolor='#4799eb')

    W = eigvecs[:, :2]
    X_train_lda = X_train @ W

    class_0 = X_train_lda[y_train == 0.0]
    class_1 = X_train_lda[y_train == 1.0]
    class_2 = X_train_lda[y_train == 2.0]

    plt.scatter(class_0[:, 0], class_0[:, 1], marker=".", color=colors[4], label="Class 0")
    plt.scatter(class_1[:, 0], class_1[:, 1], marker=".", color=colors[2], label="Class 1")
    plt.scatter(class_2[:, 0], class_2[:, 1], marker=".", color=colors[1], label="Class 2")

    plt.legend()
    plt.title("LDA Projection (2D)")
    plt.xlabel("LD1")
    plt.ylabel("LD2")
    plt.savefig("lda_projection_2D.png")
    plt.close()

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
    
    graph_2D_scatter(X_train_norm, y_train, eigvecs)
    
    def project_data(n_components):
        W = eigvecs[:, :n_components]
        W = eigvecs[:, :n_components]
        X_train_lda = X_train_norm @ W
        X_test_lda = X_test_norm @ W
        return X_train_lda, X_test_lda
    
    return project_data

# Search function for LDA
def lda_search_fn(X_train_lda, y_train, X_test_lda, y_test, get_cms=False):
    class_means_lda = {c: np.mean(X_train_lda[y_train == c], axis=0) for c in np.unique(y_train)}

    def predict(x):
        distances = {c: np.linalg.norm(x - mean) for c, mean in class_means_lda.items()}
        return min(distances, key=distances.get)

    y_pred = np.array([predict(x) for x in X_test_lda])
    if get_cms: conf_matrix(y_pred, y_test, title = "LDA Classifier Results", print_results=True)
    return accuracy_score(y_test, y_pred)

# Search function for Random Forest
def rf_search_fn(X_train, y_train, X_test, y_test, n_estimators):
    rf_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=sd)
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)
    return accuracy_score(y_test, y_pred)

def plot_accuracies(x_values, accuracies, xlabel, ylabel, title, label=None):
    #plt.figure(facecolor='#120e24ff')

    plt.plot(x_values, accuracies, marker='o', label=label, color=colors[-1])
    colors.pop()
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)

    plt.grid(True, linestyle='--', color='gray', alpha=0.6)

    if label:
        plt.legend()

def plot_accuracies_bar(rf_data):
    categories = list(rf_data.keys())
    categories = [str(key) for key in rf_data.keys()]
    values = list(rf_data.values())

    # Create the bar plot
    #plt.figure(figsize=(8, 5))
    fig = plt.figure(facecolor='#4799eb')

    plt.bar(categories, values, color=colors, edgecolor='#120e24ff')

    # Add labels and title
    plt.xlabel('Number of Components')
    plt.ylabel('Accuracy')
    plt.title('RF Accuracy With/Without LDA Reduction (n_estimators=100)')
    plt.ylim(0.75, 0.85)  # Set minimum y-axis limit for better focus
    plt.grid(axis='y', linestyle='--', alpha=0.7)


    # saving plot
    plt.savefig("accuracies_rf_bar.png")


# Function to combine LDA and Random Forest in a pipeline
def lda_rf_pipeline(X_train, y_train, X_test, y_test, n_components_list, n_estimators_list):
    rf_data = {}
    classes = np.unique(y_train)
    max_components = min(len(classes) - 1, X_train.shape[1])
    
    compute_projection = perform_lda(X_train, y_train, X_test)
    plt.figure(facecolor='#4799eb')
    lda_accuracies = []
    for n_components in n_components_list:
        print(f"Testing LDA with {n_components} components")
        X_train_lda, X_test_lda = compute_projection(n_components)
        acc = lda_search_fn(X_train_lda, y_train, X_test_lda, y_test, get_cms=True)
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
    #plt.show()


    for i, acc in enumerate(rf_no_lda_accuracies):
        print(f"RF with no LDA (n_estimators={n_estimators_list[i]}) accuracy: {acc}\n")
    
    plot_accuracies_bar(rf_data)


# Function to process data file and apply searches
def process_data_file(file_path, n_estimators_list):
    print(f"\nProcessing file: {file_path}")
    data = pd.read_csv(file_path)
    data = data.sample(frac=0.8, random_state=42) #DELETE LATER
    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values

    split_ratio = 0.8
    indices = np.random.permutation(len(X))
    split_index = int(len(X) * split_ratio)
    X_train, X_test = X[indices[:split_index]], X[indices[split_index:]]
    y_train, y_test = y[indices[:split_index]], y[indices[split_index:]]

    classes = np.unique(y_train)
    max_components = 12#min(len(classes) - 1, X_train.shape[1])
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


    