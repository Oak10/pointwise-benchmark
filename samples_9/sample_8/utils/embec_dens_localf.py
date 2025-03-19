import numpy as np

class EmbecDensLocalF:
    """
    A class to compute the pointwise reliability for predictions
    based on density and local fit (data agreement, and ML agreement)
    following [1].

    Ref:
    [1] Jorge Henriques, Teresa Rocha, Simão Paredes, Paulo Gil, João Loureiro,
    and Lorena Petrella. Pointwise reliability of machine learning models:
    Application to cardiovascular risk assessment. In Tomaž Jarm, Rok
    Šmerc, and Samo Mahnič-Kalamiza, editors, 9th European Medical and
    Biological Engineering Conference, pages 213–222, Cham, 2024. Springer
    Nature Switzerland.
    """

    def __init__(self, X_train, y_train, pipeline, k_max, threshold):
        """
        Parameters:
        - X_train: Training data (preprocessed).
        - y_train: Training labels.
        - pipeline: Trained pipeline for classification.
        - k_max: Maximum number of neighbors.
        - threshold: Distance threshold for neighbors.
        """

        self.X_train = X_train
        self.y_train = y_train
        self.pipeline = pipeline
        self.k_max = k_max
        self.threshold = threshold

    # def compute_density(self, new_instance):
    #     """
    #     Compute the density component for reliability.

    #     Parameters:
    #     - new_instance: Instance for which density is computed.

    #     Returns:
    #     - density: Density component of the reliability.
    #     """
    #     distances = np.linalg.norm(self.X_train - new_instance, axis=1)
    #     neighbors_within_threshold = np.sum(distances < self.threshold)
    #     density = neighbors_within_threshold / self.k_max if self.k_max > 0 else 0
    #     return density

    def compute_density(self, new_instance):
        """
        Compute the density component for reliability.

        Parameters:
        - new_instance: Instance for which density is computed (preprocessed).

        Returns:
        - density: Density component of the reliability.
        """
        distances = np.linalg.norm(self.X_train - new_instance, axis=1)
        neighbors_within_threshold = np.sum(distances < self.threshold)
        # Clip neighbors_within_threshold at k_max and normalize
        density = min(neighbors_within_threshold, self.k_max) / self.k_max if self.k_max > 0 else 0
        return density

    def compute_data_agreement(self, new_instance):
        """
        Compute the data agreement component for reliability.

        Parameters:
        - new_instance: Instance for which agreement is computed (preprocessed).

        Returns:
        - data_agreement: Data agreement component of the reliability.
        """
        distances = np.linalg.norm(self.X_train - new_instance, axis=1)
        neighbors_idx = np.where(distances < self.threshold)[0]

        if len(neighbors_idx) > 0:
            neighbors_labels = self.y_train[neighbors_idx]
            predicted_label = self.pipeline.named_steps["classifier"].predict([new_instance])[0]
            data_agreement = np.mean(neighbors_labels == predicted_label)
        else:
            data_agreement = 0.0
        return data_agreement

    def compute_ml_agreement(self, new_instance):
        """
        Compute the ML agreement component for reliability.

        Parameters:
        - new_instance: Instance for which agreement is computed (preprocessed)

        Returns:
        - ml_agreement: ML agreement component of the reliability.
        """
        distances = np.linalg.norm(self.X_train - new_instance, axis=1)
        neighbors_idx = np.where(distances < self.threshold)[0]

        if len(neighbors_idx) > 0:
            neighbors_predictions = self.pipeline.named_steps["classifier"].predict(self.X_train[neighbors_idx])
            predicted_label = self.pipeline.named_steps["classifier"].predict([new_instance])[0]
            ml_agreement = np.mean(neighbors_predictions == predicted_label)
        else:
            ml_agreement = 0.0
        return ml_agreement

    def compute_reliability(self, new_instance):
        """
        Compute the reliability score for a given instance.

        Parameters:
        - new_instance: The instance to evaluate (preprocessed)

        Returns:
        - reliability_score: Computed reliability score.
        """
        # Density Component
        density = self.compute_density(new_instance)

        # Data Agreement Component
        data_agreement = self.compute_data_agreement(new_instance)

        # ML Agreement Component
        ml_agreement = self.compute_ml_agreement(new_instance)

        reliability_score = density * data_agreement * ml_agreement

        # Ensure reliability is in [0, 1]
        #reliability_score = max(0.0, min(1.0, reliability_score))
        return reliability_score
 