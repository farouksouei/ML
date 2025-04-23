import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
from sklearn.pipeline import Pipeline
from datetime import datetime
import os


class UserEngagementProcessor:
    def __init__(self, file_path=None, data=None):
        """Initialize with either file path or data."""
        # Create visualization directory
        os.makedirs('charts/engagement', exist_ok=True)

        if file_path:
            self.load_data_from_file(file_path)
        elif data is not None:
            if isinstance(data, pd.DataFrame):
                self.df = data.copy()
            else:
                self.df = pd.DataFrame(data)
        else:
            self.df = pd.DataFrame()

    def load_data_from_file(self, file_path):
        """Load data from CSV file."""
        try:
            self.df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error loading data: {e}")
            self.df = pd.DataFrame()

    def clean_data(self):
        """Clean and preprocess the data."""
        # Check if the data is already loaded with column names
        if self.df.shape[0] > 0 and not all(isinstance(col, str) for col in self.df.columns):
            # Rename columns
            self.df.columns = ['user_id', 'joinedDate', 'joinedCourses', 'skills', 'desired_skills', 'isVerified']

        # Convert date strings to datetime objects
        try:
            self.df['joinedDate'] = pd.to_datetime(self.df['joinedDate'])
        except Exception as e:
            print(f"Warning: Error converting joinedDate to datetime: {e}")
            # If error, create a days_since_joined column with default values
            self.df['days_since_joined'] = 365  # Default value
        else:
            # Calculate days since joined
            self.df['days_since_joined'] = (datetime.now() - self.df['joinedDate']).dt.days

        # Convert course and skill strings to lists
        self.df['joinedCourses'] = self.df['joinedCourses'].apply(self._string_to_list)
        self.df['skills'] = self.df['skills'].apply(self._string_to_list)
        self.df['desired_skills'] = self.df['desired_skills'].apply(self._string_to_list)

        # Convert verification status to boolean if it's not already
        self.df['isVerified'] = self.df['isVerified'].map(
            lambda x: True if str(x).lower() == 'true' else False
        )

        # Add derived features
        self.df['course_count'] = self.df['joinedCourses'].apply(len)
        self.df['skill_count'] = self.df['skills'].apply(len)
        self.df['desired_skill_count'] = self.df['desired_skills'].apply(len)

        # Calculate skill gaps (desired skills not in current skills)
        self.df['skill_gap'] = self.df.apply(
            lambda x: [skill for skill in x['desired_skills'] if skill not in x['skills']],
            axis=1
        )
        self.df['skill_gap_count'] = self.df['skill_gap'].apply(len)

        # Add engagement level based on course count
        self.df['engagement_level'] = self.df['course_count'].apply(self._categorize_engagement)

        # Add popular skills presence
        self._add_popular_skill_features()

        return self

    def _add_popular_skill_features(self):
        """Add binary features for popular skills."""
        popular_skills = ['JavaScript', 'Python', 'SQL', 'HTML', 'CSS', 'Java', 'Node.js', 'AI', 'Machine Learning']

        for skill in popular_skills:
            skill_key = skill.lower().replace(".", "").replace(" ", "_")
            self.df[f'has_{skill_key}'] = self.df['skills'].apply(
                lambda skills: 1 if skill in skills else 0
            )
            self.df[f'wants_{skill_key}'] = self.df['desired_skills'].apply(
                lambda skills: 1 if skill in skills else 0
            )

    def _string_to_list(self, skills_str):
        """Convert a string of skills to a list."""
        # Handle the case when skills_str is a Series or array
        if isinstance(skills_str, (pd.Series, list, np.ndarray)):
            return [self._string_to_list(item) for item in skills_str]

        # Handle null values, empty strings, and "Unknown"
        if pd.isna(skills_str) or skills_str == "" or skills_str == "Unknown":
            return []

        # Remove quotes, brackets and split by comma
        if isinstance(skills_str, str):
            skills_str = skills_str.strip('"[]\'')
            # Split by comma and strip whitespace and quotes from each item
            skills_list = [item.strip().strip('"\'') for item in skills_str.split(',')]
            return [item for item in skills_list if item]  # Remove empty items

        return []  # Default return empty list for any other case

    def _categorize_engagement(self, course_count):
        """Categorize engagement level based on course count."""
        if course_count >= 4:
            return "Highly Engaged"
        elif course_count >= 2:
            return "Moderately Engaged"
        else:
            return "Minimally Engaged"

    def visualize_engagement_distribution(self):
        """Visualize distribution of engagement levels."""
        if 'engagement_level' not in self.df.columns:
            print("Error: engagement_level column not found. Run clean_data() first.")
            return

        plt.figure(figsize=(10, 6))
        sns.countplot(x='engagement_level', data=self.df, palette='viridis')
        plt.title('Distribution of User Engagement Levels', fontsize=14)
        plt.xlabel('Engagement Level', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.tight_layout()
        plt.savefig('charts/engagement/engagement_distribution.png')
        plt.close()

        # Also visualize relationship between features and engagement
        self._visualize_feature_relationships()

    def _visualize_feature_relationships(self):
        """Visualize relationships between features and engagement level."""
        # Select features to visualize
        features = []

        # Add features only if they exist
        if 'days_since_joined' in self.df.columns:
            features.append('days_since_joined')
        if 'course_count' in self.df.columns:
            features.append('course_count')
        if 'skill_count' in self.df.columns:
            features.append('skill_count')
        if 'desired_skill_count' in self.df.columns:
            features.append('desired_skill_count')
        if 'skill_gap_count' in self.df.columns:
            features.append('skill_gap_count')

        if not features:
            print("Warning: No features available for visualization.")
            return

        plt.figure(figsize=(15, 12))
        for i, feature in enumerate(features, 1):
            plt.subplot(len(features), 1, i)
            sns.boxplot(x='engagement_level', y=feature, data=self.df, palette='coolwarm')
            plt.title(f'{feature.replace("_", " ").title()} by Engagement Level', fontsize=12)
            plt.xlabel('Engagement Level', fontsize=10)
            plt.ylabel(feature.replace("_", " ").title(), fontsize=10)
            plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig('charts/engagement/feature_relationships.png')
        plt.close()

    def prepare_features(self):
        """Prepare features for the classification model."""
        # Select relevant features that definitely exist
        features = []

        # Check if each feature exists in the DataFrame
        potential_features = ['days_since_joined', 'skill_count', 'desired_skill_count',
                              'skill_gap_count', 'isVerified']

        for feature in potential_features:
            if feature in self.df.columns:
                features.append(feature)

        # Add popular skill features that exist
        feature_cols = [col for col in self.df.columns if col.startswith('has_') or col.startswith('wants_')]
        features.extend(feature_cols)

        if not features:
            raise ValueError("No features available for classification. Check data preprocessing.")

        # Create feature matrix
        X = self.df[features].copy()

        # Convert boolean to integer if necessary
        if 'isVerified' in X.columns and X['isVerified'].dtype == bool:
            X['isVerified'] = X['isVerified'].astype(int)

        # Target variable
        y = self.df['engagement_level']

        return X, y

    def run_knn_classifier(self, n_neighbors=5, cv_folds=5):
        """Run KNN classifier with cross-validation."""
        X, y = self.prepare_features()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        # Create pipeline with scaling
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=n_neighbors))
        ])

        # Train the model
        pipeline.fit(X_train, y_train)

        # Make predictions
        y_pred = pipeline.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )

        # Cross-validation
        cv = KFold(n_splits=min(cv_folds, len(X)), shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')

        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=pipeline.classes_,
                    yticklabels=pipeline.classes_)
        plt.title('Confusion Matrix - KNN Classifier', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.savefig('charts/engagement/knn_confusion_matrix.png')
        plt.close()

        # Find optimal K
        self._find_optimal_k(X, y, max_k=min(20, len(X) - 1))

        results = {
            'model': 'KNN',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'trained_model': pipeline
        }

        return results

    def _find_optimal_k(self, X, y, max_k=20):
        """Find the optimal value of K for KNN classifier and save plot."""
        k_range = range(1, max_k + 1)
        scores = []

        for k in k_range:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('knn', KNeighborsClassifier(n_neighbors=k))
            ])
            cv = KFold(n_splits=min(5, len(X)), shuffle=True, random_state=42)
            cv_score = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
            scores.append(cv_score.mean())

        # Plotting the K vs Accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, scores, marker='o', linestyle='-', color='green')
        plt.title('K-Value vs. Cross-Validated Accuracy')
        plt.xlabel('Number of Neighbors (K)')
        plt.ylabel('CV Accuracy')
        plt.xticks(k_range)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('charts/engagement/optimal_k_plot.png')
        plt.close()


# Example usage code
if __name__ == "__main__":
    # Sample data from your dataset7
    file_path = '../data/edited_skill_exchange_dataset.csv'
    processor = UserEngagementProcessor(file_path=file_path)
    processor.clean_data()

    # Create processor and process data
    processor.clean_data()

    # Visualize engagement distribution
    processor.visualize_engagement_distribution()

    # Run KNN classifier
    results = processor.run_knn_classifier(n_neighbors=3)

    # Print results
    print("\nClassification Results for KNN:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"Cross-validation mean accuracy: {results['cv_mean']:.4f}")
    print(f"Cross-validation std: {results['cv_std']:.4f}")