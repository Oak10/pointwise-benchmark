import numpy as np

# TODO: Subtractive clustering for categorical data https://ieeexplore.ieee.org/abstract/document/7603354
# another implementation: https://github.com/alanjeffares/subtractive-clustering/blob/master/subtractive_clustering.py
#       based -> (Approx from our implementation)Support vector machines based on subtractive clustering: https://ieeexplore.ieee.org/abstract/document/1527702


class SubtractiveClustering:
    """
    Subtractive Clustering Algorithm for finding cluster centers.

    Based on: https://www.mathworks.com/help/fuzzy/subclust.html

    Algorithms:
        Subtractive clustering assumes that each data point is a potential cluster center. The algorithm does the following:
            1 - Calculate the likelihood that each data point would define a cluster center, based on the density of surrounding data points.
            2 - Choose the data point with the highest potential to be the first cluster center.
            3 - Remove all data points near the first cluster center. The vicinity is determined using clusterInfluenceRange.
            4 - Choose the remaining point with the highest potential as the next cluster center.
            5 - Repeat steps 3 and 4 until all the data is within the influence range of a cluster center.

        The subtractive clustering method is an extension of the mountain clustering method proposed in [2].
    
    Refs:
    1. Chiu, Stephen L. “Fuzzy Model Identification Based on Cluster Estimation.” 
    Journal of Intelligent and Fuzzy Systems 2, no. 3 (1994): 267–78.
    https://doi.org/10.3233/IFS-1994-2306.

    2. Yager, Ronald R., and Dimitar P. Filev. “Generation of Fuzzy Rules by Mountain Clustering.” 
    Journal of Intelligent and Fuzzy Systems 2, no. 3 (1994): 209–19.
    https://doi.org/10.3233/IFS-1994-2301.
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

        # Compute potential for each data point
        potentials = np.zeros(n_samples)
        for i in range(n_samples):
            # https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
            distances = np.linalg.norm(data - data[i], axis=1)
            potentials[i] = np.sum(np.exp(-((distances**2) / (self.cluster_influence_range**2))))

        # Initialize list of cluster centers
        cluster_centers = []
        max_potential = np.max(potentials)
        first_center_idx = np.argmax(potentials)
        cluster_centers.append(data[first_center_idx])
        P1 = max_potential

        # Adjust potentials iteratively
        while True:
            # Suppress potential of points near the most recent cluster center
            distances = np.linalg.norm(data - cluster_centers[-1], axis=1)
            potentials -= P1 * np.exp(-((distances**2) / ((self.squash_factor * self.cluster_influence_range)**2)))

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
                # Ambiguous case, find the minimum distance to existing centers
                new_center_idx = np.argmax(potentials)
                new_point = data[new_center_idx]
                distances_to_centers = np.linalg.norm(cluster_centers - new_point, axis=1)
                dmin = np.min(distances_to_centers)

                # Accept if sufficiently far from existing centers
                if dmin / self.cluster_influence_range + max_potential / P1 >= 1:
                    cluster_centers.append(new_point)
                else:
                    # Stop if the conditions for accepting are not met
                    break

        self.cluster_centers = np.array(cluster_centers)
        return self.cluster_centers
