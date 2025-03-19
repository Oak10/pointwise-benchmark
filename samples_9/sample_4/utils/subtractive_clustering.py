import numpy as np

class SubtractiveClustering:
    """
    Subtractive Clustering Algorithm for finding cluster centers.

    Based on:
        -  https://www.mathworks.com/help/fuzzy/subclust.html
        - Chiu (1994): Fuzzy Model Identification Based on Cluster Estimation.

    Algorithm Steps:
        1. Compute the density (potential) for each data point:
            D_i = sum_{j=1}^{n} exp(-||x_i - x_j||^2 / (r_a/2)^2)
        Where:
        - r_a is the cluster influence range.

        2. Select the point with the highest potential as the first cluster center.
        3. Suppress the potential of points within the influence range (scaled by squash_factor) of the identified center.
        4. Choose the next highest potential point as the next cluster center.
          4.1 If the point's potential is between acceptance and rejection thresholds, evaluate its distance to existing centers using:
          (d_min / r_a) + (D_ck / D_1) ≥ 1.
        5. Repeat steps 3 and 4 until no points exceed the rejection threshold.

    References:
    [1] Chiu, Stephen L. "Fuzzy Model Identification Based on Cluster Estimation."
        Journal of Intelligent and Fuzzy Systems 2, no. 3 (1994): 267–278.
        https://doi.org/10.3233/IFS-1994-2306.
    """

    def __init__(self, cluster_influence_range=0.5, squash_factor=1.25, acceptance_ratio=0.5, rejection_ratio=0.15):
        """
        Initialize the clustering parameters.

        Parameters:
        - cluster_influence_range (float): Radius of influence for density calculation (default: 0.5).
        - squash_factor (float): Factor for scaling the range of influence of cluster centers (default: 1.25).
        - acceptance_ratio (float): Fraction of potential of the first center to accept another center (default: 0.5).
        - rejection_ratio (float): Fraction of potential of the first center to reject another center (default: 0.15).
        """
        self.cluster_influence_range = cluster_influence_range
        self.squash_factor = squash_factor
        self.acceptance_ratio = acceptance_ratio
        self.rejection_ratio = rejection_ratio
        self.cluster_centers = None

    def fit(self, data):
        """
        Apply subtractive clustering to the data to find cluster centers.

        Parameters:
        - data (np.array): Input data points, shape (n_samples, n_features).

        Returns:
        - cluster_centers (np.array): Array of identified cluster centers.
        """
        # Number of samples
        n_samples = data.shape[0]

        # Compute influence (r_a/2)
        influence_radius = self.cluster_influence_range / 2

        # Compute potentials
        potentials = np.zeros(n_samples)
        for i in range(n_samples):
            # https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
            distances = np.linalg.norm(data - data[i], axis=1)
            potentials[i] = np.sum(np.exp(-((distances ** 2) / (influence_radius ** 2))))

        # Initialize cluster centers
        cluster_centers = []
        max_potential = np.max(potentials)
        first_center_idx = np.argmax(potentials)
        cluster_centers.append(data[first_center_idx])
        P1 = max_potential

        # Iterative selection of clusters
        while True:
            # Suppress potentials of points near the most recently added cluster center (Eq. 2)

            distances = np.linalg.norm(data - cluster_centers[-1], axis=1)
            potentials -= P1 * np.exp(-((distances ** 2) / ((self.squash_factor * influence_radius) ** 2)))

            # Find next highest potential point
            max_potential = np.max(potentials)

            if max_potential < self.rejection_ratio * P1:
                # Stop if maximum potential is below the rejection threshold
                break
            elif max_potential >= self.acceptance_ratio * P1:
                # Accept the point as a new cluster center
                new_center_idx = np.argmax(potentials)
                cluster_centers.append(data[new_center_idx])
            else:
                # Ambiguous Case: Evaluate points with moderate potential
                # Ambiguous case, find the minimum distance to existing centers
                new_center_idx = np.argmax(potentials)
                new_point = data[new_center_idx]

                # Check distance to existing centers
                distances_to_centers = np.linalg.norm(np.array(cluster_centers) - new_point, axis=1)
                dmin = np.min(distances_to_centers)

                # Accept if sufficiently far from existing centers ((Eq. 3))
                if dmin / self.cluster_influence_range + max_potential / P1 >= 1:
                    cluster_centers.append(new_point)
                else:
                    potentials[new_center_idx] = 0  # suppress ambiguous point potential

        self.cluster_centers = np.array(cluster_centers)
        return self.cluster_centers