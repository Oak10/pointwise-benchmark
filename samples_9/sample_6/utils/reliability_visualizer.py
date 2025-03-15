import matplotlib.pyplot as plt
from matplotlib.table import Table
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE


class ReliabilityVisualizer:
    """
    Visualization of reliability scores for predictions,
    """

    def __init__(self, results_df, true_label_col='true_label', predicted_label_col='predicted_label', reliability_score_col='reliability_score'):
        """
        Parameters:
        - results_df: DataFrame containing true labels, predicted labels, and reliability scores.
        - true_label_col: Column name for the true labels.
        - predicted_label_col: Column name for the predicted labels.
        - reliability_score_col: Column name for the reliability scores.

        Inputs:
        - results_df = [
                {"true_label": true_label,"predicted_label": predicted_label,"reliability_score": reliability_score},
                (...)]
        """
        self.results_df = results_df
        self.true_label_col = true_label_col
        self.predicted_label_col = predicted_label_col
        self.reliability_score_col = reliability_score_col


    def plot_reliability_scores(self, fig_size=(12, 7)):
        """
        Scatter plot.

        Parameters:
        - fig_size: Tuple specifying the figure size as (width, height).
        """
        plt.figure(figsize=fig_size)

        for idx, row in self.results_df.iterrows():
            color = 'blue' if row[self.true_label_col] == row[self.predicted_label_col] else 'red'
            marker = 'o' if row[self.predicted_label_col] == 0 else 's'  # Circle for class 0, Squares for class 1
            plt.scatter(idx, row[self.reliability_score_col], color=color, marker=marker)

        # Define legend
        correct_class_0 = plt.Line2D([0], [0], marker='o', color='blue', markersize=7, linestyle='None', label='Correct (Class 0 - out)')
        correct_class_1 = plt.Line2D([0], [0], marker='s', color='blue', markersize=7, linestyle='None', label='Correct (Class 1 - in)')
        incorrect_class_0 = plt.Line2D([0], [0], marker='o', color='red', markersize=7, linestyle='None', label='Incorrect (Class 0 - out)')
        incorrect_class_1 = plt.Line2D([0], [0], marker='s', color='red', markersize=7, linestyle='None', label='Incorrect (Class 1 - in)')
        plt.legend(handles=[correct_class_0, correct_class_1, incorrect_class_0, incorrect_class_1], loc='lower right')

        plt.xlabel("Instance Index")
        plt.ylabel("Reliability Score")
        plt.title("Reliability Scores for Predictions")
        plt.grid(True)
        plt.show()


    def get_reliability_table_5(self):
        """
        Create a percentage-based reliability table based on corrected intervals.

        Returns:
        - DataFrame
        """

        intervals = [
            (0.0, 0.05),
            (0.05, 0.1),
            (0.1, 0.15),
            (0.15, 0.2),
            (0.2, 0.25),
            (0.25, 0.3),
            (0.3, 0.35),
            (0.35, 0.4),
            (0.4, 0.45),
            (0.45, 0.5),
            (0.5, 0.55),
            (0.55, 0.6),
            (0.6, 0.65),
            (0.65, 0.7),
            (0.7, 0.75),
            (0.75, 0.8),
            (0.8, 0.85),
            (0.85, 0.9),
            (0.9, 0.95),
            (0.95, 1.0), 
        ]

        interval_labels = [
            "[0.00, 0.05]", "]0.05, 0.10]", "]0.10, 0.15]", "]0.15, 0.20]", "]0.20, 0.25]",
            "]0.25, 0.30]", "]0.30, 0.35]", "]0.35, 0.40]", "]0.40, 0.45]", "]0.45, 0.50]",
            "]0.50, 0.55]", "]0.55, 0.60]", "]0.60, 0.65]", "]0.65, 0.70]", "]0.70, 0.75]",
            "]0.75, 0.80]", "]0.80, 0.85]", "]0.85, 0.90]", "]0.90, 0.95]", "]0.95, 1.00]"
        ]

        # Prepare the summary table structure
        table_data = {
            "Reliability": interval_labels,
            "Correct (0)": [],
            "Incorrect (0)": [],
            "Correct (1)": [],
            "Incorrect (1)": [],
        }

        results_df = self.results_df
        for i, (start, end) in enumerate(intervals):
            if i == 0:
                # First interval: closed on both sides
                subset = results_df[
                    (results_df["reliability_score"] >= start) & (results_df["reliability_score"] <= end)
                ]
            else:
                # Other intervals: open on the lower bound, closed on the upper bound
                subset = results_df[
                    (results_df["reliability_score"] > start) & (results_df["reliability_score"] <= end)
                ]

            correct_class_0 = len(
                subset[
                    (subset["true_label"] == 0) & (subset["predicted_label"] == 0)
                ]
            )
            incorrect_class_1 = len(
                subset[
                    (subset["true_label"] == 0) & (subset["predicted_label"] == 1)
                ]
            )
            correct_class_1 = len(
                subset[
                    (subset["true_label"] == 1) & (subset["predicted_label"] == 1)
                ]
            )
            incorrect_class_0 = len(
                subset[
                    (subset["true_label"] == 1) & (subset["predicted_label"] == 0)
                ]
            )

            table_data["Correct (0)"].append(correct_class_0)
            table_data["Incorrect (0)"].append(incorrect_class_0)
            table_data["Correct (1)"].append(correct_class_1)
            table_data["Incorrect (1)"].append(incorrect_class_1)

        reliability_table_df = pd.DataFrame(table_data)
        return reliability_table_df


    def get_reliability_table_10(self):
        """
        Create a percentage-based reliability table based on corrected intervals (10 %).

        Returns:
        - DataFrame
        """

        intervals = [
            (0.0, 0.1),
            (0.1, 0.2),
            (0.2, 0.3),
            (0.3, 0.4),
            (0.4, 0.5),
            (0.5, 0.6),
            (0.6, 0.7),
            (0.7, 0.8),
            (0.8, 0.9),
            (0.9, 1.0),
        ]

        interval_labels = [
            "[0.00, 0.10]", "]0.10, 0.20]", "]0.20, 0.30]",
            "]0.30, 0.40]", "]0.40, 0.50]", "]0.50, 0.60]",
            "]0.60, 0.70]", "]0.70, 0.80]", "]0.80, 0.90]",
            "]0.90, 1.00]"
        ]

        # Prepare the summary table structure
        table_data = {
            "Reliability": interval_labels,
            "Correct (0)": [],
            "Incorrect (0)": [],
            "Correct (1)": [],
            "Incorrect (1)": [],
        }

        results_df = self.results_df
        for i, (start, end) in enumerate(intervals):
            if i == 0:
                # First interval: closed on both sides
                subset = results_df[
                    (results_df["reliability_score"] >= start) & (results_df["reliability_score"] <= end)
                ]
            else:
                # Other intervals: open on the lower bound, closed on the upper bound
                subset = results_df[
                    (results_df["reliability_score"] > start) & (results_df["reliability_score"] <= end)
                ]

            correct_class_0 = len(
                subset[
                    (subset["true_label"] == 0) & (subset["predicted_label"] == 0)
                ]
            )
            incorrect_class_1 = len(
                subset[
                    (subset["true_label"] == 0) & (subset["predicted_label"] == 1)
                ]
            )
            correct_class_1 = len(
                subset[
                    (subset["true_label"] == 1) & (subset["predicted_label"] == 1)
                ]
            )
            incorrect_class_0 = len(
                subset[
                    (subset["true_label"] == 1) & (subset["predicted_label"] == 0)
                ]
            )

            table_data["Correct (0)"].append(correct_class_0)
            table_data["Incorrect (0)"].append(incorrect_class_0)
            table_data["Correct (1)"].append(correct_class_1)
            table_data["Incorrect (1)"].append(incorrect_class_1)

        reliability_table_df = pd.DataFrame(table_data)
        return reliability_table_df


    def plot_table_graph(self, df):
            # Plotting
            fig, ax = plt.subplots(figsize=(10, 6))
            width = 0.2  # width of bars
            x = range(len(df))

            # Bar plots for each category
            ax.bar(x, df["Correct (0)"], width, label="Correct (0)", color="blue", alpha=0.7)
            ax.bar([p + width for p in x], df["Incorrect (0)"], width, label="Incorrect (0)", color="red", alpha=0.7)
            ax.bar([p + 2*width for p in x], df["Correct (1)"], width, label="Correct (1)", color="green", alpha=0.7)
            ax.bar([p + 3*width for p in x], df["Incorrect (1)"], width, label="Incorrect (1)", color="orange", alpha=0.7)

            # Adding labels and legend
            ax.set_xticks([p + 1.5 * width for p in x])
            ax.set_xticklabels(df["Reliability"], rotation=45, ha="right")
            ax.set_xlabel("Reliability Intervals")
            ax.set_ylabel("Count")
            ax.set_title("Reliability Distribution by Prediction Categories")
            ax.legend()

            # Adjust layout and display
            plt.tight_layout()
            plt.show()
    
    def plot_table_image(self, df):
        fig, ax = plt.subplots(figsize=(10, 6))

        # Hide the axes
        ax.axis('tight')
        ax.axis('off')

        # Create the table
        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc='center',
            loc='center'
        )

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(df.columns))))

        # Display the table
        plt.show()

    def plot_population_and_error_rate_by_class(self, df):
        """
        Overlay population distribution and error rate for reliability intervals, separated by class (0 and 1).

        Parameters:
        - df: DataFrame with reliability intervals and counts for each prediction category.
        """
        # Calculate total predictions and error rate for each class
        df["Total Class 0 Predictions"] = df["Correct (0)"] + df["Incorrect (0)"]
        df["Total Class 1 Predictions"] = df["Correct (1)"] + df["Incorrect (1)"]
        df["Class 0 Error Rate"] = df["Incorrect (0)"] / df["Total Class 0 Predictions"]
        df["Class 1 Error Rate"] = df["Incorrect (1)"] / df["Total Class 1 Predictions"]

        fig, ax1 = plt.subplots(figsize=(12, 7))

        # Bar plot for class 0 and class 1 population
        bar_width = 0.35
        x = np.arange(len(df))
        ax1.bar(x - bar_width / 2, df["Total Class 0 Predictions"], width=bar_width, color="skyblue", alpha=0.6, label="Class 0 Population")
        ax1.bar(x + bar_width / 2, df["Total Class 1 Predictions"], width=bar_width, color="lightgreen", alpha=0.6, label="Class 1 Population")
        ax1.set_xlabel("Reliability Intervals")
        ax1.set_ylabel("Population Count", color="black")
        ax1.set_xticks(x)
        ax1.set_xticklabels(df["Reliability"], rotation=45, ha="right")
        ax1.tick_params(axis='y', labelcolor="black")

        # Overlay line plot for error rates
        ax2 = ax1.twinx()
        ax2.plot(x, df["Class 0 Error Rate"], color="blue", marker="o", label="Class 0 Error Rate")
        ax2.plot(x, df["Class 1 Error Rate"], color="green", marker="o", label="Class 1 Error Rate")
        ax2.set_ylabel("Error Rate", color="black")
        ax2.tick_params(axis='y', labelcolor="black")

        # Adding legend
        lines, labels = ax2.get_legend_handles_labels()
        bars, bar_labels = ax1.get_legend_handles_labels()
        ax1.legend(bars + lines, bar_labels + labels, loc="upper right")

        # Title and layout
        fig.suptitle("Population Distribution and Error Rate by Class Across Reliability Intervals")
        plt.tight_layout()
        plt.show()



# #######################################  T-SNE

    def plot_tsne_with_reliability(self, X_train_preprocessed, validation_data, y_train, pipeline, fig_size=(14, 10)):
        """
        Create a t-SNE plot showing reliability intervals.

        Parameters:
        - X_train_preprocessed: Preprocessed training data.
        - validation_data: Original validation data.
        - y_train: True labels for the training data.
        - pipeline: The trained machine learning pipeline.
        - fig_size: Tuple specifying the figure size.
        """

        # Define reliability intervals
        intervals = np.arange(0, 1.05, 0.1)
        self.results_df['reliability_interval'] = pd.cut(
            self.results_df['reliability_score'], bins=intervals, include_lowest=True
        )

        # Generate a color palette
        palette = sns.color_palette("coolwarm", len(intervals) - 1)
        interval_colors = {interval: palette[i] for i, interval in enumerate(self.results_df['reliability_interval'].cat.categories)}
       
        ## Just check
        # print("Reliability Interval to Color Mapping:")
        # for interval, color in interval_colors.items():
        #     print(f"Interval: {interval}, Color: {color}")
        # Verify reliability interval assignment
        print("Sample reliability interval assignment:")
        print(self.results_df[['reliability_score', 'reliability_interval']].head(10))

        # Preprocess validation data
        validation_data_features = validation_data.drop(columns=['SOURCE'])
        validation_data_preprocessed = pipeline.named_steps['preprocessor'].transform(validation_data_features)

        # Combine training and validation data for t-SNE
        combined_data = np.vstack((X_train_preprocessed, validation_data_preprocessed))


        # Apply t-SNE
        tsne_combined = TSNE(n_components=2, perplexity=30, random_state=42)
        reduced_combined_data = tsne_combined.fit_transform(combined_data)

        # Split into training and validation
        reduced_training_data = reduced_combined_data[:len(X_train_preprocessed)]
        reduced_validation_data = reduced_combined_data[len(X_train_preprocessed):]

        # Add t-SNE results to results_df
        self.results_df['TSNE1'] = reduced_validation_data[:, 0]
        self.results_df['TSNE2'] = reduced_validation_data[:, 1]

        # Create training DataFrame for visualization
        training_df = pd.DataFrame(reduced_training_data, columns=['TSNE1', 'TSNE2'])
        training_df['true_label'] = y_train

        # Visualization
        plt.figure(figsize=fig_size)


        # Plot training data
        for _, row in training_df.iterrows():
            marker = 's' if row['true_label'] == 0 else 'o'  # Square for class 0, Circle for class 1
            plt.scatter(
                row['TSNE1'], row['TSNE2'],
                facecolors='none',  # No fill color
                edgecolors='black',  # Black outline
                marker=marker,
                s=30,
                linewidth=0.8
            )

        # Plot validation data with reliability intervals
        for _, row in self.results_df.iterrows():
            marker = 's' if row['predicted_label'] == 0 else 'o'
            plt.scatter(
                row['TSNE1'], row['TSNE2'],
                facecolors=interval_colors[row['reliability_interval']],
                edgecolors=interval_colors[row['reliability_interval']],
                marker=marker,
                s=30,
                linewidth=1,
                alpha=0.6  # Transparency
            )


        # Legend
        interval_legend_handles = [
            plt.Line2D([0], [0], marker='o', color=color, label=f"{interval.left:.2f}-{interval.right:.2f}", markersize=8, linestyle='')
            for interval, color in interval_colors.items()
        ]
        data_type_handles = [
            plt.Line2D([0], [0], marker='s', color='black', label='Training Data (Class 0)', markersize=8, linestyle='', markerfacecolor='none'),
            plt.Line2D([0], [0], marker='o', color='black', label='Training Data (Class 1)', markersize=8, linestyle='', markerfacecolor='none'),
            plt.Line2D([0], [0], marker='s', color='black', label='Validation Data (Class 0)', markersize=8, linestyle='', markerfacecolor='black'),
            plt.Line2D([0], [0], marker='o', color='black', label='Validation Data (Class 1)', markersize=8, linestyle='', markerfacecolor='black')
        ]
        plt.legend(
            handles=interval_legend_handles + data_type_handles,
            loc='upper left',
            title="Legend"
        )

        plt.title("t-SNE Visualization with Reliability Scores")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.tight_layout()
        plt.show()

