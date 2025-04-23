import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings

# Add parent directory to path for imports
sys.path.append('..')

from DataProcessor import DataProcessor
from PredictNewCourses.KNN import SkillRecommender


def test_skill_recommender():
    # Suppress sklearn warnings about F-score
    warnings.filterwarnings("ignore", category=UserWarning)

    print("=== Testing KNN Skill Recommender ===")

    # Initialize DataProcessor and load data
    processor = DataProcessor(file_path='../data/edited_skill_exchange_dataset.csv', columns=6)
    processor.df.columns = ["id", "enrollment_date", "current_skills",
                            "desired_skills", "target_skills", "success"]
    processor.clean_data()

    print(f"Loaded dataset with {len(processor.df)} entries")

    # Initialize and prepare recommender
    recommender = SkillRecommender(processor)
    recommender.prepare_data(use_pca=True, pca_components=0.95)

    # Find optimal k
    print("\nFinding optimal k...")
    optimal_k = recommender.find_optimal_k(k_range=range(1, 10, 2))

    # Train model with optimal k
    print(f"\nTraining model with k={optimal_k}...")
    recommender.train_model(n_neighbors=5)

    # Evaluate model
    results = recommender.evaluate_model()

    # Display metrics
    print("\n=== Model Performance Metrics ===")
    print(f"Accuracy:        {results['accuracy']:.4f}")
    print(f"Precision (Micro): {results['precision_micro']:.4f}")
    print(f"Recall (Micro):    {results['recall_micro']:.4f}")
    print(f"F1 Score (Micro):  {results['f1_micro']:.4f}")
    print(f"F1 Score (Macro):  {results['f1_macro']:.4f}")

    # Test with a few sample profiles
    test_profiles = [
        {"name": "Web Developer", "skills": ["HTML", "CSS", "JavaScript"]},
        {"name": "Data Scientist", "skills": ["Python", "Statistics", "Machine Learning"]},
        {"name": "DevOps Engineer", "skills": ["Linux", "Docker", "AWS"]}
    ]

    print("\n=== Sample Recommendations ===")
    for profile in test_profiles:
        print(f"\nRecommendations for {profile['name']}:")
        try:
            recommendations = recommender.recommend_skills(profile['skills'], top_n=3)
            for skill, score in recommendations:
                print(f"- {skill} (confidence: {score:.4f})")
        except Exception as e:
            print(f"Error: {e}")

    return recommender, results


if __name__ == "__main__":
    # Ensure output directories exist
    os.makedirs('charts/skill_recommender', exist_ok=True)

    # Run test
    recommender, metrics = test_skill_recommender()

    # Create a summary metrics visualization
    plt.figure(figsize=(10, 6))
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 (Micro)', 'F1 (Macro)']
    metrics_values = [
        metrics['accuracy'],
        metrics['precision_micro'],
        metrics['recall_micro'],
        metrics['f1_micro'],
        metrics['f1_macro']
    ]

    bars = plt.bar(metrics_names, metrics_values, color='skyblue')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom')

    plt.ylim(0, 1.1)
    plt.title('KNN Skill Recommender Performance')
    plt.ylabel('Score')
    plt.savefig('charts/skill_recommender/performance_summary.png')
    plt.close()

    print("\n=== Testing Complete ===")