import numpy as np
from sklearn.neighbors import NearestNeighbors

class ICMFramework:
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
        Select the k-nearest neighbors of the query point from the memory.

        Parameters:
        - query_point (np.array): The point for which neighbors are selected.

        Returns:
        - selected_cases (list): The k-nearest neighbors as (x, y) pairs.
        """

        X_memory = np.array([x for x, _ in self.memory])
        y_memory = np.array([y for _, y in self.memory])

        nbrs = NearestNeighbors(n_neighbors=min(self.k_neighbors, len(self.memory)))
        nbrs.fit(X_memory)
        distances, indices = nbrs.kneighbors([query_point])

        selected_cases = [(X_memory[idx], y_memory[idx]) for idx in indices[0]]
        return selected_cases

    def compute_confidence(self, query_point, predicted_label):
        """
        Compute the confidence score for a query point.

        Parameters:
        - query_point (np.array): The point for which the confidence is calculated.
        - predicted_label (int): The label predicted by the classifier.

        Returns:
        - confidence (float): The confidence score for the query point.
        """
        # Select neighbors
        selected_cases = self.select_neighbors(query_point)

        # Separate S+ and S-
        S_plus = [x for x, y in selected_cases if y == predicted_label]
        S_minus = [x for x, y in selected_cases if y != predicted_label]

        # Define the weighting function (e.g., Gaussian kernel)
        def weight(x1, x2):
            # TODO: Re use the distances calculated @ select_neighbors
            distance = np.linalg.norm(x1 - x2)  # Euclidean  || x1 - x2 ||
            # https://www.sciencedirect.com/topics/veterinary-science-and-veterinary-medicine/radial-basis-function 
            # e^(-distance²)  given scale parameter equals to1 (σ=1²)
            return np.exp(-distance**2)  

        # Weighted sum
        weight_S_plus = sum(weight(query_point, x_i) for x_i in S_plus)
        weight_S_minus = sum(weight(query_point, x_i) for x_i in S_minus)

        # Compute normalization factor Z(x)
        Z_x = 1 / (weight_S_plus + weight_S_minus) if (weight_S_plus + weight_S_minus) > 0 else 1

        # Compute confidence [ -1, 1]
        raw_confidence = Z_x * (weight_S_plus - weight_S_minus)     # -> ( weight_S_plus − weight_S_minus ) / (weight_S_plus + weight_S_minus)  

        # Rescale confidence to range [0, 1]
        # rescaled_confidence = (raw_confidence + 1) / 2
        # confidence to range [0, 1]
        if raw_confidence < 0:
            rescaled_confidence = 0
        else:
            rescaled_confidence = raw_confidence

        return rescaled_confidence
