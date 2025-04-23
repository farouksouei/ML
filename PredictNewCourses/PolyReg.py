import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import time

# Add parent directory to path for imports
sys.path.append('..')

from DataProcessor import DataProcessor
from PredictNewCourses.KNN import SkillRecommender


def test_polynomial_regression_recommender():
    """Test and evaluate Polynomial Regression for skill recommendation"""
    # Suppress sklearn warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    print("=== Testing Polynomial Regression Skill Recommender ===")

    # Initialize DataProcessor and load data
    processor = DataProcessor(file_path='../data/edited_skill_exchange_dataset.csv', columns=6)
    processor.df.columns = ["id", "enrollment_date", "current_skills",
                            "desired_skills", "target_skills", "success"]
    processor.clean_data()

    print(f"Loaded dataset with {len(processor.df)} entries")

    # Initialize recommender for data preparation
    base_recommender = SkillRecommender(processor)
    base_recommender.prepare_data(use_pca=False)  # Use raw features for polynomial expansion

    # Test different polynomial degrees
    degrees = [1, 2, 3, 4]
    results = {}
    training_times = {}
    prediction_times = {}

    for degree in degrees:
        print(f"\nTesting Polynomial Regression with degree={degree}...")

        # Train Polynomial Regression model
        start_time = time.time()
        models = []
        for i in range(base_recommender.y_train.shape[1]):
            # Create a pipeline for each target skill
            polynomial_model = Pipeline([
                ('poly', PolynomialFeatures(degree=degree)),
                ('linear', LinearRegression())
            ])
            # Train on the i-th target skill
            polynomial_model.fit(base_recommender.X_train, base_recommender.y_train[:, i])
            models.append(polynomial_model)

        training_times[degree] = time.time() - start_time
        print(f"Training time: {training_times[degree]:.2f} seconds")

        # Evaluate model
        start_time = time.time()
        y_pred_prob = np.zeros(base_recommender.y_test.shape)
        for i, model in enumerate(models):
            y_pred_prob[:, i] = model.predict(base_recommender.X_test)

        # Convert probabilities to binary predictions using 0.5 threshold
        y_pred = (y_pred_prob >= 0.5).astype(int)

        # Clip predictions to [0, 1] range for valid probabilities
        y_pred_prob = np.clip(y_pred_prob, 0, 1)

        prediction_times[degree] = time.time() - start_time
        print(f"Prediction time: {prediction_times[degree]:.2f} seconds")

        # Calculate metrics with zero_division parameter
        results[degree] = {
            'accuracy': (base_recommender.y_test == y_pred).mean(),
            'precision_micro': precision_score(base_recommender.y_test, y_pred, average='micro', zero_division=0),
            'recall_micro': recall_score(base_recommender.y_test, y_pred, average='micro', zero_division=0),
            'f1_micro': f1_score(base_recommender.y_test, y_pred, average='micro', zero_division=0),
            'f1_macro': f1_score(base_recommender.y_test, y_pred, average='macro', zero_division=0)
        }

        # Display metrics
        print("\n=== Model Performance Metrics ===")
        for metric, value in results[degree].items():
            print(f"{metric}: {value:.4f}")

    # Create comparison visualization
    visualize_poly_comparison(results, training_times, prediction_times)

    # Test with the best degree
    best_degree = max(results.keys(), key=lambda k: results[k]['f1_micro'])
    print(f"\nBest polynomial degree based on F1-micro score: {best_degree}")

    # Test with sample profiles using the best model
    test_sample_profiles(base_recommender, best_degree)

    # Save results to CSV
    df_results = pd.DataFrame(results).T
    os.makedirs('charts/PolynomialRegression/skill_recommender', exist_ok=True)
    df_results.to_csv('charts/PolynomialRegression/skill_recommender/polynomial_results.csv')

    return results


def visualize_poly_comparison(results, training_times, prediction_times):
    """Create visualization comparing different polynomial degrees"""
    # Create directory for Polynomial Regression results
    os.makedirs('charts/PolynomialRegression/skill_recommender', exist_ok=True)

    # Extract metrics for comparison
    degrees = list(results.keys())
    metrics = ['accuracy', 'precision_micro', 'recall_micro', 'f1_micro', 'f1_macro']

    # Performance metrics comparison
    plt.figure(figsize=(14, 8))

    # Prepare data for grouped bar chart
    bar_width = 0.15
    positions = np.arange(len(degrees))

    for i, metric in enumerate(metrics):
        values = [results[degree][metric] for degree in degrees]
        plt.bar(positions + i * bar_width, values, width=bar_width,
                label=metric.capitalize().replace('_', ' '))

    plt.xlabel('Polynomial Degree')
    plt.ylabel('Score')
    plt.title('Polynomial Regression Performance with Different Degrees')
    plt.xticks(positions + bar_width * 2, degrees)
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('charts/PolynomialRegression/skill_recommender/degree_performance_comparison.png')
    plt.close()

    # Computation time comparison
    plt.figure(figsize=(10, 6))

    # Prepare data for grouped bar chart
    bar_width = 0.35
    positions = np.arange(len(degrees))

    plt.bar(positions - bar_width / 2, [training_times[d] for d in degrees], width=bar_width,
            label='Training Time', color='skyblue')
    plt.bar(positions + bar_width / 2, [prediction_times[d] for d in degrees], width=bar_width,
            label='Prediction Time', color='salmon')

    plt.xlabel('Polynomial Degree')
    plt.ylabel('Time (seconds)')
    plt.title('Computation Time for Different Polynomial Degrees')
    plt.xticks(positions, degrees)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('charts/PolynomialRegression/skill_recommender/degree_time_comparison.png')
    plt.close()

    # Create a radar chart for metric comparison
    plt.figure(figsize=(10, 8))

    # Prepare data for radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    plt.subplot(polar=True)

    colors = sns.color_palette('viridis', len(degrees))

    for i, degree in enumerate(degrees):
        values = [results[degree][metric] for metric in metrics]
        values += values[:1]  # Close the loop

        plt.plot(angles, values, '-', linewidth=2, color=colors[i], label=f"Degree {degree}")
        plt.fill(angles, values, alpha=0.1, color=colors[i])

    plt.xticks(angles[:-1], [m.capitalize().replace('_', ' ') for m in metrics])
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], color='gray')
    plt.ylim(0, 1)
    plt.title('Performance Metrics by Polynomial Degree')
    plt.legend(loc='upper right')
    plt.savefig('charts/PolynomialRegression/skill_recommender/degree_radar_comparison.png')
    plt.close()

    # Create a heatmap for detailed metric comparison
    plt.figure(figsize=(12, 8))

    # Prepare data for heatmap
    heatmap_data = []
    for degree in degrees:
        row = [results[degree][metric] for metric in metrics]
        heatmap_data.append(row)

    df_heatmap = pd.DataFrame(heatmap_data, index=[f"Degree {d}" for d in degrees],
                              columns=[m.capitalize().replace('_', ' ') for m in metrics])

    sns.heatmap(df_heatmap, annot=True, cmap='viridis', vmin=0, vmax=1, linewidths=0.5, fmt='.3f')
    plt.title('Detailed Polynomial Regression Performance Comparison')
    plt.tight_layout()
    plt.savefig('charts/PolynomialRegression/skill_recommender/degree_heatmap_comparison.png')
    plt.close()

    # Learning curves comparison
    plt.figure(figsize=(12, 8))
    plt.plot(degrees, [results[d]['f1_micro'] for d in degrees], 'o-', label='F1 Micro')
    plt.plot(degrees, [results[d]['precision_micro'] for d in degrees], 's-', label='Precision')
    plt.plot(degrees, [results[d]['recall_micro'] for d in degrees], '^-', label='Recall')
    plt.plot(degrees, [results[d]['accuracy'] for d in degrees], 'd-', label='Accuracy')

    plt.xlabel('Polynomial Degree')
    plt.ylabel('Score')
    plt.title('Learning Curve by Polynomial Degree')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('charts/PolynomialRegression/skill_recommender/learning_curve.png')
    plt.close()


def test_sample_profiles(base_recommender, best_degree):
    """Test model with sample user profiles using polynomial regression"""
    # Train models with the best degree
    models = []
    for i in range(base_recommender.y_train.shape[1]):
        polynomial_model = Pipeline([
            ('poly', PolynomialFeatures(degree=best_degree)),
            ('linear', LinearRegression())
        ])
        polynomial_model.fit(base_recommender.X_train, base_recommender.y_train[:, i])
        models.append(polynomial_model)

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

        # Get predictions from all models
        skill_scores = {}
        for i, model in enumerate(models):
            skill_name = base_recommender.target_names[i]
            if skill_name not in profile['skills']:  # Don't recommend skills they already have
                pred = model.predict(user_features)[0]
                # Clip prediction to [0, 1] range
                pred = max(0, min(1, pred))
                skill_scores[skill_name] = pred

        # Sort and get top recommendations
        recommendations = sorted(skill_scores.items(), key=lambda x: x[1], reverse=True)[:3]

        for skill, score in recommendations:
            print(f"- {skill} (confidence: {score:.4f})")

        # Visualize recommendations
        visualize_recommendations(profile, recommendations)


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
    plt.title(f'Polynomial Regression Recommendations for {profile["name"]}')
    plt.xlabel('Recommendation Confidence')
    plt.tight_layout()

    save_path = f'charts/PolynomialRegression/skill_recommender/recommendations_{profile["name"].lower().replace(" ", "_")}.png'
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    # Run test
    results = test_polynomial_regression_recommender()
    print("\n=== Polynomial Regression Testing Complete ===")