import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import time

# Add parent directory to path for imports
sys.path.append('..')

from DataProcessor import DataProcessor
from PredictNewCourses.KNN import SkillRecommender

def test_decision_tree_recommender():
    """Test and evaluate decision tree for skill recommendation"""
    # Suppress sklearn warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    print("=== Testing Decision Tree Skill Recommender ===")

    # Initialize DataProcessor and load data
    processor = DataProcessor(file_path='../data/edited_skill_exchange_dataset.csv', columns=6)
    processor.df.columns = ["id", "enrollment_date", "current_skills",
                           "desired_skills", "target_skills", "success"]
    processor.clean_data()

    print(f"Loaded dataset with {len(processor.df)} entries")

    # Initialize recommender for data preparation
    base_recommender = SkillRecommender(processor)
    base_recommender.prepare_data(use_pca=False)  # Decision trees handle raw features better

    # Test different max_depth values
    depths = [3, 5, 7, 10, None]
    results = {}
    training_times = {}
    prediction_times = {}

    for depth in depths:
        depth_name = str(depth) if depth is not None else "unlimited"
        print(f"\nTesting Decision Tree with max_depth={depth_name}...")

        # Train Decision Tree model
        start_time = time.time()
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        model = MultiOutputClassifier(dt)
        model.fit(base_recommender.X_train, base_recommender.y_train)
        training_times[depth_name] = time.time() - start_time
        print(f"Training time: {training_times[depth_name]:.2f} seconds")

        # Evaluate model
        start_time = time.time()
        y_pred = model.predict(base_recommender.X_test)
        prediction_times[depth_name] = time.time() - start_time
        print(f"Prediction time: {prediction_times[depth_name]:.2f} seconds")

        # Calculate metrics with zero_division parameter
        results[depth_name] = {
            'accuracy': (base_recommender.y_test == y_pred).mean(),
            'precision_micro': precision_score(base_recommender.y_test, y_pred, average='micro', zero_division=0),
            'recall_micro': recall_score(base_recommender.y_test, y_pred, average='micro', zero_division=0),
            'f1_micro': f1_score(base_recommender.y_test, y_pred, average='micro', zero_division=0),
            'f1_macro': f1_score(base_recommender.y_test, y_pred, average='macro', zero_division=0)
        }

        # Display metrics
        print("\n=== Model Performance Metrics ===")
        for metric, value in results[depth_name].items():
            print(f"{metric}: {value:.4f}")

    # Create comparison visualization
    visualize_dt_comparison(results, training_times, prediction_times)

    # Test with the best depth
    best_depth = max(results.keys(), key=lambda k: results[k]['f1_micro'])
    print(f"\nBest max_depth based on F1-micro score: {best_depth}")

    # Get the actual depth value for model creation
    best_depth_value = None if best_depth == "unlimited" else int(best_depth)

    # Visualize the tree structure
    visualize_tree_structure(base_recommender, best_depth_value)

    # Test with sample profiles using the best model
    test_sample_profiles(base_recommender, MultiOutputClassifier(
        DecisionTreeClassifier(max_depth=best_depth_value, random_state=42)
    ))

    return results

def visualize_tree_structure(recommender, max_depth):
    """Visualize the structure of the first decision tree"""
    # Create directories
    os.makedirs('charts/DecisionTree/skill_recommender', exist_ok=True)

    # Only visualize if we're using raw features (not PCA)
    if recommender.pca is None:
        # Train a single decision tree for the first target
        dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)

        # Get the first target to visualize (assuming multi-label)
        y_first_target = recommender.y_train[:, 0]

        # Train the tree
        dt.fit(recommender.X_train, y_first_target)

        # Get feature names
        feature_names = recommender.feature_names

        plt.figure(figsize=(20, 10))
        plot_tree(dt, filled=True, feature_names=feature_names,
                 class_names=['No', 'Yes'], rounded=True, fontsize=8)
        plt.title(f"Decision Tree Structure (First Target, Depth={max_depth})")
        plt.tight_layout()
        plt.savefig('charts/DecisionTree/skill_recommender/tree_structure.png', dpi=300)
        plt.close()

        # Print feature importances
        importances = dt.feature_importances_
        indices = np.argsort(importances)[-10:]  # Get top 10 features

        plt.figure(figsize=(10, 8))
        plt.title('Top 10 Feature Importances')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig('charts/DecisionTree/skill_recommender/feature_importance.png')
        plt.close()

def visualize_dt_comparison(results, training_times, prediction_times):
    """Create visualization comparing different decision tree depths"""
    # Create directory for Decision Tree results
    os.makedirs('charts/DecisionTree/skill_recommender', exist_ok=True)

    # Extract metrics for comparison
    depths = list(results.keys())
    metrics = ['accuracy', 'precision_micro', 'recall_micro', 'f1_micro', 'f1_macro']

    # Performance metrics comparison
    plt.figure(figsize=(14, 8))

    # Prepare data for grouped bar chart
    bar_width = 0.15
    positions = np.arange(len(depths))

    for i, metric in enumerate(metrics):
        values = [results[depth][metric] for depth in depths]
        plt.bar(positions + i * bar_width, values, width=bar_width,
                label=metric.capitalize().replace('_', ' '))

    plt.xlabel('Max Tree Depth')
    plt.ylabel('Score')
    plt.title('Decision Tree Performance at Different Depths')
    plt.xticks(positions + bar_width * 2, depths)
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('charts/DecisionTree/skill_recommender/depth_performance_comparison.png')
    plt.close()

    # Computation time comparison
    plt.figure(figsize=(10, 6))

    # Prepare data for grouped bar chart
    bar_width = 0.35
    positions = np.arange(len(depths))

    plt.bar(positions - bar_width/2, [training_times[d] for d in depths], width=bar_width,
            label='Training Time', color='skyblue')
    plt.bar(positions + bar_width/2, [prediction_times[d] for d in depths], width=bar_width,
            label='Prediction Time', color='salmon')

    plt.xlabel('Max Tree Depth')
    plt.ylabel('Time (seconds)')
    plt.title('Computation Time for Different Tree Depths')
    plt.xticks(positions, depths)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('charts/DecisionTree/skill_recommender/depth_time_comparison.png')
    plt.close()

    # Create a radar chart for metric comparison
    plt.figure(figsize=(10, 8))

    # Prepare data for radar chart
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    plt.subplot(polar=True)

    colors = sns.color_palette('viridis', len(depths))

    for i, depth in enumerate(depths):
        values = [results[depth][metric] for metric in metrics]
        values += values[:1]  # Close the loop

        plt.plot(angles, values, '-', linewidth=2, color=colors[i], label=depth)
        plt.fill(angles, values, alpha=0.1, color=colors[i])

    plt.xticks(angles[:-1], [m.capitalize().replace('_', ' ') for m in metrics])
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], color='gray')
    plt.ylim(0, 1)
    plt.title('Performance Metrics by Tree Depth')
    plt.legend(loc='upper right')
    plt.savefig('charts/DecisionTree/skill_recommender/depth_radar_comparison.png')
    plt.close()

    # Create a heatmap for detailed metric comparison
    plt.figure(figsize=(12, 8))

    # Prepare data for heatmap
    heatmap_data = []
    for depth in depths:
        row = [results[depth][metric] for metric in metrics]
        heatmap_data.append(row)

    df_heatmap = pd.DataFrame(heatmap_data, index=depths, columns=[m.capitalize().replace('_', ' ') for m in metrics])

    sns.heatmap(df_heatmap, annot=True, cmap='viridis', vmin=0, vmax=1, linewidths=0.5, fmt='.3f')
    plt.title('Detailed Performance Comparison')
    plt.tight_layout()
    plt.savefig('charts/DecisionTree/skill_recommender/depth_heatmap_comparison.png')
    plt.close()

def test_sample_profiles(base_recommender, model):
    """Test model with sample user profiles"""
    # Train the model with best depth
    model.fit(base_recommender.X_train, base_recommender.y_train)

    # Test with a few sample profiles
    test_profiles = [
        {"name": "Web Developer", "skills": ["HTML", "CSS", "JavaScript"]},
        {"name": "Data Scientist", "skills": ["Python", "Statistics", "Machine Learning"]},
        {"name": "DevOps Engineer", "skills": ["Linux", "Docker", "AWS"]}
    ]

    print("\n=== Sample Recommendations ===")
    for profile in test_profiles:
        print(f"\nRecommendations for {profile['name']}:")

        # Transform input skills to feature vector
        user_features = np.zeros((1, len(base_recommender.feature_names)))
        for skill in profile['skills']:
            if skill in base_recommender.feature_names:
                idx = np.where(base_recommender.feature_names == skill)[0]
                if len(idx) > 0:
                    user_features[0, idx[0]] = 1

        # Apply same preprocessing as during training
        user_features = base_recommender.scaler.transform(user_features)

        # Apply PCA if it was used in training
        if base_recommender.pca is not None:
            user_features = base_recommender.pca.transform(user_features)

        # Get predictions
        try:
            y_pred = model.predict(user_features)

            # For recommendations based on highest confidence
            skill_scores = {}
            for i, pred in enumerate(y_pred[0]):
                skill_name = base_recommender.target_names[i]
                if skill_name not in profile['skills']:
                    skill_scores[skill_name] = float(pred)  # Convert to float for consistent handling

            # Sort and get top recommendations
            recommendations = sorted(skill_scores.items(), key=lambda x: x[1], reverse=True)[:3]

            for skill, score in recommendations:
                print(f"- {skill} (confidence: {score:.4f})")

            # Visualize recommendations
            visualize_recommendations(profile, recommendations)

        except Exception as e:
            print(f"Error: {e}")

def visualize_recommendations(profile, recommendations):
    """Create a simple horizontal bar chart of recommendations."""
    if not recommendations:
        return

    plt.figure(figsize=(10, 6))
    skills = [r[0] for r in recommendations]
    scores = [r[1] for r in recommendations]

    bars = plt.barh(skills, scores, color='lightgreen')

    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                 f'{width:.3f}', ha='left', va='center')

    plt.xlim(0, 1.1)
    plt.title(f'Decision Tree Recommendations for {profile["name"]}')
    plt.xlabel('Recommendation Confidence')
    plt.tight_layout()

    plt.savefig(f'charts/DecisionTree/skill_recommender/recommendations_{profile["name"].lower().replace(" ", "_")}.png')
    plt.close()

def compare_all_models():
    """Compare Decision Tree, SVM, and KNN models"""
    # Create directory for comparisons
    os.makedirs('charts/model_comparison', exist_ok=True)

    # Placeholder for loading results from different algorithms
    # In a real implementation, you would load saved results
    # For now, we'll create some sample data

    model_results = {
        'KNN': {
            'accuracy': 0.83,
            'precision_micro': 0.82,
            'recall_micro': 0.79,
            'f1_micro': 0.80,
            'f1_macro': 0.72,
            'training_time': 0.18,
            'prediction_time': 0.22
        },
        'SVM (RBF)': {
            'accuracy': 0.85,
            'precision_micro': 0.84,
            'recall_micro': 0.81,
            'f1_micro': 0.82,
            'f1_macro': 0.75,
            'training_time': 3.25,
            'prediction_time': 0.45
        },
        'Decision Tree': {
            'accuracy': 0.79,
            'precision_micro': 0.78,
            'recall_micro': 0.77,
            'f1_micro': 0.77,
            'f1_macro': 0.70,
            'training_time': 0.15,
            'prediction_time': 0.05
        }
    }

    # Create bar chart comparing key metrics
    plt.figure(figsize=(14, 8))

    models = list(model_results.keys())
    metrics = ['accuracy', 'precision_micro', 'recall_micro', 'f1_micro', 'f1_macro']

    # Prepare data for grouped bar chart
    bar_width = 0.15
    positions = np.arange(len(models))

    for i, metric in enumerate(metrics):
        values = [model_results[model][metric] for model in models]
        plt.bar(positions + i * bar_width, values, width=bar_width,
                label=metric.capitalize().replace('_', ' '))

    plt.xlabel('Model Type')
    plt.ylabel('Score')
    plt.title('Skill Recommender Performance Comparison')
    plt.xticks(positions + bar_width * 2, models)
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('charts/model_comparison/model_performance_comparison.png')
    plt.close()

    # Create time comparison
    plt.figure(figsize=(10, 6))

    # Prepare data for grouped bar chart
    bar_width = 0.35
    positions = np.arange(len(models))

    plt.bar(positions - bar_width/2, [model_results[m]['training_time'] for m in models], width=bar_width,
            label='Training Time', color='skyblue')
    plt.bar(positions + bar_width/2, [model_results[m]['prediction_time'] for m in models], width=bar_width,
            label='Prediction Time', color='salmon')

    plt.xlabel('Model Type')
    plt.ylabel('Time (seconds)')
    plt.title('Computation Time Comparison')
    plt.xticks(positions, models)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('charts/model_comparison/model_time_comparison.png')
    plt.close()

if __name__ == "__main__":
    # Run test
    results = test_decision_tree_recommender()

    # Optionally, compare with other models
    # compare_all_models()

    print("\n=== Decision Tree Testing Complete ===")