import numpy as np
from sklearn.neighbors import NearestNeighbors

class ICMFrameworkV2:
    # Adapted  (Interpretable Confidence Measures) from [1]
    #         Parameters:
    #         - memory_size (int): Maximum size of the memory D.
    #         - k_neighbors (int): Number of nearest neighbors to use in the selection step.
    #
    # Diffs: use static D, return scalled [0;1]
    #       
    #         [1] Jasper van der Waa, Tjeerd Schoonderwoerd, Jurriaan van Diggelen, and
    #         Mark Neerincx. Interpretable confidence measures for decision support
    #         systems. International Journal of Human-Computer Studies, 144:102493, 2020
    def __init__(self, training_data, training_labels, k_neighbors=10):
        """
        Calculates a confidence score for a given query point based on the support or opposition of its nearest neighbors
        Parameters:
        - training_data (np.array): Array of instances from the training dataset.
        - training_labels (np.array): Labels of the training dataset.
        - k_neighbors (int): Number of nearest neighbors to use in the selection step.

        How:
            - Takes an instance and its predicted label from the classifier
            - Identifies the k nearest neighbors (Gaussian) from the training data
                - Separates neighbors into those supporting (S+) and opposing (S-) the predicted label
        
        Reliability Calculation:
            - Computes a weighted score based on the balance of support (S+) and opposition (S-).
        Normalizes the reliability [0,1]. Being 1 high confidence
        """
        if len(training_data) != len(training_labels):
            raise ValueError("Training data and labels must have the same length.")

        self.memory = list(zip(training_data, training_labels))
        self.k_neighbors = k_neighbors

    def select_neighbors(self, query_point):
        """
        Select the k-nearest neighbors for the query point.

        Parameters:
        - query_point (np.array): Query point.

        Returns:
        - selected_cases (list): List of (neighbor, label) pairs.
        """
        X_memory = np.array([x for x, _ in self.memory])
        y_memory = np.array([y for _, y in self.memory])

        nbrs = NearestNeighbors(n_neighbors=min(self.k_neighbors, len(self.memory)))
        nbrs.fit(X_memory)
        distances, indices = nbrs.kneighbors([query_point])

        selected_cases = [(X_memory[idx], y_memory[idx]) for idx in indices[0]]
        return selected_cases

    def compute_sigma(self, query_point, neighbors):
        """
        Compute sigma as the mean squared distance between the query point and its neighbors.

        Parameters:
        - query_point (np.array): Query point.
        - neighbors (list of np.array): List of neighbor points.

        Returns:
        - sigma (float): Mean squared distance.
        """
        sigma = np.mean([np.linalg.norm(query_point - x_i) ** 2 for x_i in neighbors])
        print("Sigma:")
        print(sigma)

        return sigma

    def weight(self, x1, x2, sigma):
        """
        Compute the weight between two points using the "ICM-3 formula"

        Parameters:
        - x1 (np.array): First point.
        - x2 (np.array): Second point.
        - sigma (float): Scaling parameter (mean squared distance).

        Returns:
        - weight (float): Weight based on the formula.
        """
        distance_squared = np.linalg.norm(x1 - x2) ** 2
        fraction = distance_squared / sigma
        return np.exp(-fraction ** 2)

    def compute_confidence(self, query_point, predicted_label):
        """
        Compute the ICM-3 confidence score for a query point.

        Parameters:
        - query_point (np.array): Query point.
        - predicted_label (int): Predicted label.

        Returns:
        - confidence (float): Confidence score.
        """
        selected_cases = self.select_neighbors(query_point)

        # Separate S+ and S-
        S_plus = [x for x, y in selected_cases if y == predicted_label]
        S_minus = [x for x, y in selected_cases if y != predicted_label]
        all_neighbors = [x for x, _ in selected_cases]  # Combine all neighbors for sigma

        # Compute sigma (mean squared distance)
        sigma = self.compute_sigma(query_point, all_neighbors)

        # Compute weighted contributions for S+ and S-
        if len(S_plus) > 0:
            weighted_sum_S_plus = sum(self.weight(query_point, x_i, sigma) for x_i in S_plus) / len(S_plus)
        else:
            weighted_sum_S_plus = 0

        if len(S_minus) > 0:
            weighted_sum_S_minus = sum(self.weight(query_point, x_j, sigma) for x_j in S_minus) / len(S_minus)
        else:
            weighted_sum_S_minus = 0

        # Compute final confidence
        confidence = weighted_sum_S_plus - weighted_sum_S_minus

        # range [0, 1]
        if confidence < 0:
            rescaled_confidence = 0
        else:
            rescaled_confidence = confidence

        return rescaled_confidence

        # return confidence
