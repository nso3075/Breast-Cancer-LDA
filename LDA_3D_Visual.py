import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import joblib


# load LDA results
X_train_lda = np.load("lda_results/X_train_lda.npy")
y_train = np.load("lda_results/y_train.npy")
X_test_lda = np.load("lda_results/X_test_lda.npy")
y_test = np.load("lda_results/y_test.npy")

# Load LDA + RF results
X_train_lda_rf = np.load("lda_results/X_train_lda_rf.npy")
y_train_rf = np.load("lda_results/y_train_rf.npy")
X_test_lda_rf = np.load("lda_results/X_test_lda_rf.npy")
y_test_rf = np.load("lda_results/y_test_rf.npy")
y_pred_rf = np.load("lda_results/y_pred_rf.npy")
rf_classifier = joblib.load("lda_results/rf_classifier.pkl")

# LDA Classifier (Euclidean Distance to Class Means)
def lda_classifier(grid_points):
    distances = [
        np.linalg.norm(grid_points - mean, axis=1) for mean in np.unique(y_train)
    ]
    return np.hstack([grid_points, np.argmin(distances, axis=0).reshape(-1, 1)])

# LDA + RF Classifier
def lda_rf_classifier(grid_points):
    class_scores = rf_classifier.predict_proba(grid_points)
    return np.hstack([grid_points, class_scores.argmax(axis=1).reshape(-1, 1)])

# Generate 3D Contours
def draw_contours(data_matrix, class_fn, title, file_name):
    # Determine the range for the plot
    x_min, x_max = data_matrix[:, 0].min() - 1, data_matrix[:, 0].max() + 1
    y_min, y_max = data_matrix[:, 1].min() - 1, data_matrix[:, 1].max() + 1

    # Generate a grid of points
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Compute the decision boundaries
    Z = class_fn(grid_points)[:, -1].reshape(xx.shape)

    # Create the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the decision boundary as a surface
    ax.plot_surface(xx, yy, Z, cmap='coolwarm', edgecolor='none', alpha=0.7)

    # Create scatter plot for individual data points
    classes = np.unique(data_matrix[:, -1]) # Get unique class labels
    for cls in classes:
        # Filter the points belonging to current class
        cls_points = data_matrix[data_matrix[:, -1] == cls]

        # Plot the points below the decision surface
        ax.scatter(
            cls_points[:, 0], # X coordinates
            cls_points[:, 1], # Y coordinates
            np.full(cls_points.shape[0], cls), # Z coordinate
            label = f"Class {int(cls)}", # Label for the legend
            s = 20, # Marker size
        )
    # Add title and legend to plot
    ax.set_title(title)
    ax.legend()

    # Save the plot as an img
    plt.savefig(file_name)
    # Display the plot
    plt.show()

# Generate Contours for Train and Test Sets
draw_contours(
    np.column_stack((X_train_lda, y_train)),
    lda_classifier,
    "LDA Train",
    "lda_train_3d.png",
)
draw_contours(
    np.column_stack((X_test_lda, y_test)),
    lda_classifier,
    "LDA Test",
    "lda_test_3d.png",
)
draw_contours(
    np.column_stack((X_train_lda_rf, y_train_rf)),
    lda_rf_classifier,
    "LDA + RF Train",
    "lda_rf_train_3d.png",
)
draw_contours(
    np.column_stack((X_test_lda_rf, y_test_rf)),
    lda_rf_classifier,
    "LDA + RF Test",
    "lda_rf_test_3d.png",
)