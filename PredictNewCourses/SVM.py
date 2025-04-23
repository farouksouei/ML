import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import cross_val_score
import seaborn as sns
import time

# Add parent directory to path for imports
sys.path.append('..')

from DataProcessor import DataProcessor
from PredictNewCourses.KNN import SkillRecommender


def test_svm_skill_recommender():
    # Suppress sklearn warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    print("=== Testing SVM Skill Recommender ===")

    # Initialize DataProcessor and load data
    processor = DataProcessor(file_path='../data/edited_skill_exchange_dataset.csv', columns=6)
    processor.df.columns = ["id", "enrollment_date", "current_skills",
                            "desired_skills", "target_skills", "success"]
    processor.clean_data()

    print(f"Loaded dataset with {len(processor.df)} entries")

    # Initialize recommender for data preparation
    base_recommender = SkillRecommender(processor)
    base_recommender.prepare_data(use_pca=True, pca_components=0.95)

    # Test different kernel options
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    results = {}
    training_times = {}
    prediction_times = {}

    for kernel in kernels:
        print(f"\nTesting SVM with {kernel} kernel...")

        # Train SVM model
        start_time = time.time()
        svm = SVC(kernel=kernel, probability=True, C=1.0, random_state=42)
        model = MultiOutputClassifier(svm)
        model.fit(base_recommender.X_train, base_recommender.y_train)
        training_times[kernel] = time.time() - start_time
        print(f"Training time: {training_times[kernel]:.2f} seconds")

        # Evaluate model
        start_time = time.time()
        y_pred = model.predict(base_recommender.X_test)
        prediction_times[kernel] = time.time() - start_time
        print(f"Prediction time: {prediction_times[kernel]:.2f} seconds")

        # Calculate metrics with zero_division parameter
        results[kernel] = {
            'accuracy': accuracy_score(base_recommender.y_test, y_pred),
            'precision_micro': precision_score(base_recommender.y_test, y_pred, average='micro', zero_division=0),
            'recall_micro': recall_score(base_recommender.y_test, y_pred, average='micro', zero_division=0),
            'f1_micro': f1_score(base_recommender.y_test, y_pred, average='micro', zero_division=0),
            'f1_macro': f1_score(base_recommender.y_test, y_pred, average='macro', zero_division=0)
        }

        # Display metrics
        print("\n=== Model Performance Metrics ===")
        for metric, value in results[kernel].items():
            print(f"{metric}: {value:.4f}")

    # Create comparison visualization
    visualize_svm_comparison(results, training_times, prediction_times)

    # Test with the best kernel
    best_kernel = max(results.keys(), key=lambda k: results[k]['f1_micro'])
    print(f"\nBest kernel based on F1-micro score: {best_kernel}")

    # Test with sample profiles using the best model
    test_sample_profiles(base_recommender, MultiOutputClassifier(SVC(kernel=best_kernel, probability=True)))

    return results


def accuracy_score(y_true, y_pred):
    """Calculate accuracy for multi-label classification"""
    return (y_true == y_pred).mean()


def precision_score(y_true, y_pred, average='micro', zero_division=0):
    from sklearn.metrics import precision_score
    return precision_score(y_true, y_pred, average=average, zero_division=zero_division)


def recall_score(y_true, y_pred, average='micro', zero_division=0):
    from sklearn.metrics import recall_score
    return recall_score(y_true, y_pred, average=average, zero_division=zero_division)


def f1_score(y_true, y_pred, average='micro', zero_division=0):
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average=average, zero_division=zero_division)


def visualize_svm_comparison(results, training_times, prediction_times):
    """Create visualization comparing different SVM kernels"""
    # Create directory for SVM results
    os.makedirs('charts/SVM/skill_recommender', exist_ok=True)

    # Extract metrics for comparison
    kernels = list(results.keys())
    metrics = ['accuracy', 'precision_micro', 'recall_micro', 'f1_micro', 'f1_macro']

    # Performance metrics comparison
    plt.figure(figsize=(14, 8))

    # Prepare data for grouped bar chart
    bar_width = 0.15
    positions = np.arange(len(kernels))

    for i, metric in enumerate(metrics):
        values = [results[kernel][metric] for kernel in kernels]
        plt.bar(positions + i * bar_width, values, width=bar_width,
                label=metric.capitalize().replace('_', ' '))

    plt.xlabel('Kernel Type')
    plt.ylabel('Score')
    plt.title('SVM Performance with Different Kernels')
    plt.xticks(positions + bar_width * 2, kernels)
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('charts/SVM/skill_recommender/kernel_performance_comparison.png')
    plt.close()

    # Computation time comparison
    plt.figure(figsize=(10, 6))

    # Prepare data for grouped bar chart
    bar_width = 0.35
    positions = np.arange(len(kernels))

    plt.bar(positions - bar_width / 2, [training_times[k] for k in kernels], width=bar_width,
            label='Training Time', color='skyblue')
    plt.bar(positions + bar_width / 2, [prediction_times[k] for k in kernels], width=bar_width,
            label='Prediction Time', color='salmon')

    plt.xlabel('Kernel Type')
    plt.ylabel('Time (seconds)')
    plt.title('Computation Time for Different SVM Kernels')
    plt.xticks(positions, kernels)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('charts/SVM/skill_recommender/kernel_time_comparison.png')
    plt.close()

    # Create a radar chart for metric comparison
    plt.figure(figsize=(10, 8))

    # Prepare data for radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    plt.subplot(polar=True)

    colors = sns.color_palette('viridis', len(kernels))

    for i, kernel in enumerate(kernels):
        values = [results[kernel][metric] for metric in metrics]
        values += values[:1]  # Close the loop

        plt.plot(angles, values, '-', linewidth=2, color=colors[i], label=kernel)
        plt.fill(angles, values, alpha=0.1, color=colors[i])

    plt.xticks(angles[:-1], [m.capitalize().replace('_', ' ') for m in metrics])
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], color='gray')
    plt.ylim(0, 1)
    plt.title('Performance Metrics by Kernel Type')
    plt.legend(loc='upper right')
    plt.savefig('charts/SVM/skill_recommender/kernel_radar_comparison.png')
    plt.close()

    # Create a heatmap for detailed metric comparison
    plt.figure(figsize=(12, 8))

    # Prepare data for heatmap
    heatmap_data = []
    for kernel in kernels:
        row = [results[kernel][metric] for metric in metrics]
        heatmap_data.append(row)

    df_heatmap = pd.DataFrame(heatmap_data, index=kernels, columns=[m.capitalize().replace('_', ' ') for m in metrics])

    sns.heatmap(df_heatmap, annot=True, cmap='viridis', vmin=0, vmax=1, linewidths=0.5, fmt='.3f')
    plt.title('Detailed Performance Comparison')
    plt.tight_layout()
    plt.savefig('charts/SVM/skill_recommender/kernel_heatmap_comparison.png')
    plt.close()


def test_sample_profiles(base_recommender, model):
    """Test model with sample user profiles"""
    # Train the model with best kernel
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
            y_pred_proba = model.predict_proba(user_features)

            # Combine probabilities for each skill
            skill_scores = {}
            for i, estimator_proba in enumerate(y_pred_proba):
                skill_name = base_recommender.target_names[i]
                if skill_name not in profile['skills']:
                    skill_scores[skill_name] = estimator_proba[0][1]  # Probability of class 1

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
        plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2.,
                 f'{width:.3f}', ha='left', va='center')

    plt.xlim(0, 1.1)
    plt.title(f'SVM Recommendations for {profile["name"]}')
    plt.xlabel('Recommendation Confidence')
    plt.tight_layout()

    plt.savefig(f'charts/SVM/skill_recommender/recommendations_{profile["name"].lower().replace(" ", "_")}.png')
    plt.close()


if __name__ == "__main__":
    # Run test
    results = test_svm_skill_recommender()

    print("\n=== SVM Testing Complete ===")