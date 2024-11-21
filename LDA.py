import numpy as np

class LDA:
    def fit(self, X, y):
        # Fit the LDA to training data
        # Param: X: the feature matrix (samples, features)
        # Param: y: ground truth (samples)

        features = X.shape[1] # total numb of features
        classes = np.unique(y) # total number of class labels
        total_mean = np.mean(X, axis=0) # mean of all data points

        # dictionaries to store class mean and prior
        self.class_mean = {}
        self.priors = {}

        # Init Scatter Matrix
        ScatterWithin = np.zeros((features, features))
        ScatterBetween = np.zeros((features, features))

        for c in classes:
            # Find all data points in class c
            X_c = X[y == c]

            # Find the mean vector of class c
            mean_c = np.mean(X_c, axis = 0)
            self.class_mean[c] = mean_c

            # Find the prior for c 
            self.priors[c] = X_c.shape[0] / X.shape[0]

            # Update the scatter matrix
            ScatterWithin += (X_c - total_mean).reshape(-1, 1)

            mean_diff = (mean_c - total_mean).reshape(-1, 1)
            ScatterBetween += X_c.shape[0] * mean_diff.dot(mean_diff.T)

        # Eigenvalue stuff -- look over, chatgpt helped here
        eigvals, eigvecs = np.linalg.eig(np.linalg.inv(ScatterWithin).dot(ScatterBetween))
        
        # Sort the eingenvectors
        sorted_indices = np.argsort(-eigvals.real)
        eigvals = eigvals[sorted_indices]
        eigvecs = eigvecs[:, sorted_indices]

        # Select the top component eigenvecs
        ## if self.components


    def predict(self, X):
        # Classify new points based on learned data
        # Param: X: feature matrix of new data points (samples, features)
        # Return: predicted class labels (samples)

        predictions = []

        for x in X:
            posteriors = []
            for c, mean_c in self.class_mean.items():
                # calc the likelihood
                # exp (-1/2 * (x-mu)^T * (x-mu))
                likelihood = np.exp()