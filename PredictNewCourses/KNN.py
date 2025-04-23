from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import warnings


class SkillRecommender:
    def __init__(self, data_processor):
        """Initialize the skill recommender with processed data."""
        self.df = data_processor.df
        self.mlb = MultiLabelBinarizer()
        self.scaler = StandardScaler()
        self.pca = None
        self.knn_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.target_names = None

        # Create directories for visualization outputs
        os.makedirs('charts/skill_recommender', exist_ok=True)

    def prepare_data(self, test_size=0.2, random_state=42, use_pca=True, pca_components=0.95):
        """
        Prepare data for training with preprocessing steps.

        Parameters:
        -----------
        test_size: float, default=0.2
            Proportion of dataset to be used as test set
        random_state: int, default=42
            Random seed for reproducibility
        use_pca: bool, default=True
            Whether to use PCA for dimensionality reduction
        pca_components: float or int, default=0.95
            Number of components or variance ratio to keep
        """
        # Transform current skills into binary features
        X = self.mlb.fit_transform(self.df['current_skills'])
        self.feature_names = self.mlb.classes_

        # Transform desired skills into binary features
        y_mlb = MultiLabelBinarizer()
        y = y_mlb.fit_transform(self.df['desired_skills'])
        self.target_names = y_mlb.classes_

        # Normalize features
        X = self.scaler.fit_transform(X)

        # Apply dimensionality reduction if requested
        if use_pca and X.shape[1] > 5:
            self.pca = PCA(n_components=pca_components, random_state=random_state)
            X = self.pca.fit_transform(X)
            explained_var = np.sum(self.pca.explained_variance_ratio_)
            n_components = self.pca.n_components_
            print(f"PCA: Using {n_components} components explaining {explained_var:.2%} of variance")

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y.sum(axis=1) if y.shape[0] > 50 else None
        )

        print(f"Feature matrix shape after preprocessing: {self.X_train.shape}")
        print(f"Target matrix shape: {self.y_train.shape}")
        print(f"Number of distinct skills as targets: {len(self.target_names)}")

        return self

    def find_optimal_k(self, k_range=range(1, 11, 2)):
        """Find optimal number of neighbors using cross-validation."""
        # Use a smaller range of k values for simplicity
        k_scores = []
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(knn, self.X_train, self.y_train, cv=5, scoring='f1_samples')
            k_scores.append(scores.mean())

        # Find the optimal K value
        optimal_k = k_range[np.argmax(k_scores)]

        # Visualize K optimization
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, k_scores, marker='o')
        plt.title('K Value Optimization')
        plt.xlabel('Number of Neighbors (k)')
        plt.ylabel('F1 Score (mean)')
        plt.axvline(x=optimal_k, color='red', linestyle='--')
        plt.text(optimal_k + 0.5, min(k_scores), f'Optimal k={optimal_k}', color='red')
        plt.grid(True)
        plt.savefig('charts/skill_recommender/k_optimization.png')
        plt.close()

        print(f"Optimal number of neighbors: {optimal_k}")
        return optimal_k

    def train_model(self, n_neighbors=5):
        """Train KNN model with distance weighting for better performance."""
        self.knn_model = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
        self.knn_model.fit(self.X_train, self.y_train)
        print(f"Model trained with {n_neighbors} neighbors and distance weighting")
        return self

    def evaluate_model(self):
        """Evaluate the trained model with zero_division parameter to handle warnings."""
        if self.knn_model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        y_pred = self.knn_model.predict(self.X_test)

        # Calculate metrics with zero_division parameter
        results = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision_micro': precision_score(self.y_test, y_pred, average='micro', zero_division=0),
            'recall_micro': recall_score(self.y_test, y_pred, average='micro', zero_division=0),
            'f1_micro': f1_score(self.y_test, y_pred, average='micro', zero_division=0),
            'f1_macro': f1_score(self.y_test, y_pred, average='macro', zero_division=0)
        }

        # Simple visualization of results
        self._visualize_evaluation(results)

        return results

    def _visualize_evaluation(self, results):
        """Create a simple bar chart of evaluation metrics."""
        metrics = ['accuracy', 'precision_micro', 'recall_micro', 'f1_micro', 'f1_macro']
        values = [results[metric] for metric in metrics]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, values, color=sns.color_palette('viridis', len(metrics)))

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                     f'{height:.3f}', ha='center', va='bottom')

        plt.ylim(0, 1.1)
        plt.title('Model Performance Metrics')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('charts/skill_recommender/model_metrics.png')
        plt.close()

    def recommend_skills(self, current_skills, top_n=5):
        """Recommend new skills based on current skills."""
        if self.knn_model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        # Transform input skills to feature vector
        user_features = np.zeros((1, len(self.feature_names)))
        for skill in current_skills:
            if skill in self.feature_names:
                idx = np.where(self.feature_names == skill)[0]
                if len(idx) > 0:
                    user_features[0, idx[0]] = 1

        # Apply same preprocessing as during training
        user_features = self.scaler.transform(user_features)

        # Apply PCA if it was used in training
        if self.pca is not None:
            user_features = self.pca.transform(user_features)

        # Predict probability for each skill
        # Using predict_proba instead of manually calculating weights
        if hasattr(self.knn_model, 'predict_proba'):
            proba = self.knn_model.predict_proba(user_features)
            skill_scores = {skill: score[0][1] for skill, score in zip(self.target_names, proba)}
        else:
            # Get neighbors and their labels
            distances, indices = self.knn_model.kneighbors(user_features)

            # Calculate weighted vote for each skill
            skill_scores = {}
            total_weight = np.sum(1 / (distances[0] + 1e-5))

            for i, neighbor_idx in enumerate(indices[0]):
                weight = 1 / (distances[0][i] + 1e-5)  # Avoid division by zero
                neighbor_skills = self.y_train[neighbor_idx]

                for j, has_skill in enumerate(neighbor_skills):
                    if has_skill:
                        skill_name = self.target_names[j]
                        if skill_name not in current_skills:
                            if skill_name in skill_scores:
                                skill_scores[skill_name] += weight / total_weight
                            else:
                                skill_scores[skill_name] = weight / total_weight

        # Filter out already known skills and sort
        recommendations = [(skill, score) for skill, score in skill_scores.items()
                           if skill not in current_skills]
        recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:top_n]

        # Visualize recommendations
        self.visualize_recommendations(current_skills, recommendations)

        return recommendations

    def visualize_recommendations(self, current_skills, recommendations):
        """Create a simple horizontal bar chart of recommendations."""
        if not recommendations:
            return

        plt.figure(figsize=(10, 6))
        skills = [r[0] for r in recommendations]
        scores = [r[1] for r in recommendations]

        bars = plt.barh(skills, scores, color='skyblue')

        # Add value labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2.,
                     f'{width:.3f}', ha='left', va='center')

        plt.xlim(0, 1.1)
        plt.title(f'Recommended Skills Based on {len(current_skills)} Current Skills')
        plt.xlabel('Recommendation Confidence')
        plt.tight_layout()

        plt.savefig(f'charts/skill_recommender/recommendations_{"-".join(current_skills)[:30]}.png')
        plt.close()