import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import xgboost as xgb
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import time

# Add parent directory to path for imports
sys.path.append('..')

from DataProcessor import DataProcessor
from PredictNewCourses.KNN import SkillRecommender
import pickle


def test_xgboost_recommender():
    """Test and evaluate XGBoost for skill recommendation"""
    # Suppress sklearn warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    print("=== Testing XGBoost Skill Recommender ===")

    # Initialize DataProcessor and load data
    processor = DataProcessor(file_path='../data/edited_skill_exchange_dataset.csv', columns=8)
    processor.df.columns = ["id", "enrollment_date", "current_skills",
                            "desired_skills", "target_skills", "success", "engagement_level", "peer_user_id"]

    # Override clean_data to work with the correct column names
    # Convert skills strings to lists
    for col in ['current_skills', 'desired_skills', 'target_skills']:
        processor.df[col] = processor.df[col].apply(processor._string_to_list)

    # Convert success to boolean
    processor.df['success'] = processor.df['success'].astype(bool)

    # Add additional derived features
    processor.df['skill_count_current'] = processor.df['current_skills'].apply(len)
    processor.df['skill_count_desired'] = processor.df['desired_skills'].apply(len)
    processor.df['skill_count_target'] = processor.df['target_skills'].apply(len)

    # Calculate skill gaps (target skills not in current)
    processor.df['skill_gap'] = processor.df.apply(
        lambda x: [skill for skill in x['target_skills'] if skill not in x['current_skills']],
        axis=1
    )
    processor.df['skill_gap_count'] = processor.df['skill_gap'].apply(len)

    # Calculate skill overlap between current and target
    processor.df['skill_overlap'] = processor.df.apply(
        lambda x: [skill for skill in x['target_skills'] if skill in x['current_skills']],
        axis=1
    )
    processor.df['skill_overlap_count'] = processor.df['skill_overlap'].apply(len)

    # Calculate learning efficiency (overlap count / target count)
    processor.df['learning_efficiency'] = processor.df.apply(
        lambda x: x['skill_overlap_count'] / len(x['target_skills']) if len(x['target_skills']) > 0 else 0,
        axis=1
    )

    print(f"Loaded dataset with {len(processor.df)} entries")

    # Rest of the function remains the same
    base_recommender = SkillRecommender(processor)
    base_recommender.prepare_data(use_pca=False)

    # Test different max_depth values
    depths = [3, 5, 7, 10]
    results = {}
    training_times = {}
    prediction_times = {}

    for depth in depths:
        print(f"\nTesting XGBoost with max_depth={depth}...")

        # Train XGBoost model
        start_time = time.time()
        xgb_model = xgb.XGBClassifier(
            max_depth=depth,
            learning_rate=0.1,
            n_estimators=100,
            objective='binary:logistic',
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model = MultiOutputClassifier(xgb_model)
        model.fit(base_recommender.X_train, base_recommender.y_train)
        training_times[depth] = time.time() - start_time
        print(f"Training time: {training_times[depth]:.2f} seconds")

        # Evaluate model
        start_time = time.time()
        y_pred = model.predict(base_recommender.X_test)
        prediction_times[depth] = time.time() - start_time
        print(f"Prediction time: {prediction_times[depth]:.2f} seconds")

        # Calculate metrics with zero_division parameter
        results[depth] = {
            'accuracy': (base_recommender.y_test == y_pred).mean(),
            'precision_micro': precision_score(base_recommender.y_test, y_pred, average='micro', zero_division=0),
            'recall_micro': recall_score(base_recommender.y_test, y_pred, average='micro', zero_division=0),
            'f1_micro': f1_score(base_recommender.y_test, y_pred, average='micro', zero_division=0),
            'f1_macro': f1_score(base_recommender.y_test, y_pred, average='macro', zero_division=0)
        }

        # Display metrics
        print("\n=== Model Performance Metrics ===")
        for metric, value in results[depth].items():
            print(f"{metric}: {value:.4f}")

    # Create comparison visualization
    visualize_xgb_comparison(results, training_times, prediction_times)

    # Test with the best depth
    best_depth = max(results.keys(), key=lambda k: results[k]['f1_micro'])
    print(f"\nBest max_depth based on F1-micro score: {best_depth}")

    # Visualize feature importance
    visualize_feature_importance(base_recommender, best_depth)

    # Test with sample profiles using the best model
    test_sample_profiles(base_recommender, MultiOutputClassifier(
        xgb.XGBClassifier(
            max_depth=best_depth,
            learning_rate=0.1,
            n_estimators=100,
            objective='binary:logistic',
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    ))

    return results,model


def visualize_feature_importance(recommender, max_depth):
    """Visualize feature importance from XGBoost"""
    # Create directories
    os.makedirs('charts/XGBoost/skill_recommender', exist_ok=True)

    # Only visualize if we're using raw features (not PCA)
    if recommender.pca is None:
        # Train XGBoost for the first target
        xgb_model = xgb.XGBClassifier(
            max_depth=max_depth,
            learning_rate=0.1,
            n_estimators=100,
            objective='binary:logistic',
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )

        # Get the first target to visualize (assuming multi-label)
        y_first_target = recommender.y_train[:, 0]

        # Train the model
        xgb_model.fit(recommender.X_train, y_first_target)

        # Get feature importance
        importance = xgb_model.feature_importances_

        # Plot top 15 features
        indices = np.argsort(importance)[-15:]

        plt.figure(figsize=(12, 10))
        plt.title(f'Top 15 Feature Importance (XGBoost, depth={max_depth})')
        plt.barh(range(len(indices)), importance[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [recommender.feature_names[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig('charts/XGBoost/skill_recommender/feature_importance.png')
        plt.close()


def visualize_xgb_comparison(results, training_times, prediction_times):
    """Create visualization comparing different XGBoost depths"""
    # Create directory for XGBoost results
    os.makedirs('charts/XGBoost/skill_recommender', exist_ok=True)

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
    plt.title('XGBoost Performance at Different Depths')
    plt.xticks(positions + bar_width * 2, depths)
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('charts/XGBoost/skill_recommender/depth_performance_comparison.png')
    plt.close()

    # Computation time comparison
    plt.figure(figsize=(10, 6))

    # Prepare data for grouped bar chart
    bar_width = 0.35
    positions = np.arange(len(depths))

    plt.bar(positions - bar_width / 2, [training_times[d] for d in depths], width=bar_width,
            label='Training Time', color='skyblue')
    plt.bar(positions + bar_width / 2, [prediction_times[d] for d in depths], width=bar_width,
            label='Prediction Time', color='salmon')

    plt.xlabel('Max Tree Depth')
    plt.ylabel('Time (seconds)')
    plt.title('Computation Time for Different XGBoost Depths')
    plt.xticks(positions, depths)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('charts/XGBoost/skill_recommender/depth_time_comparison.png')
    plt.close()

    # Create a radar chart for metric comparison
    plt.figure(figsize=(10, 8))

    # Prepare data for radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    plt.subplot(polar=True)

    colors = sns.color_palette('viridis', len(depths))

    for i, depth in enumerate(depths):
        values = [results[depth][metric] for metric in metrics]
        values += values[:1]  # Close the loop

        plt.plot(angles, values, '-', linewidth=2, color=colors[i], label=f"Depth {depth}")
        plt.fill(angles, values, alpha=0.1, color=colors[i])

    plt.xticks(angles[:-1], [m.capitalize().replace('_', ' ') for m in metrics])
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], color='gray')
    plt.ylim(0, 1)
    plt.title('Performance Metrics by XGBoost Depth')
    plt.legend(loc='upper right')
    plt.savefig('charts/XGBoost/skill_recommender/depth_radar_comparison.png')
    plt.close()

    # Create a heatmap for detailed metric comparison
    plt.figure(figsize=(12, 8))

    # Prepare data for heatmap
    heatmap_data = []
    for depth in depths:
        row = [results[depth][metric] for metric in metrics]
        heatmap_data.append(row)

    df_heatmap = pd.DataFrame(heatmap_data, index=[f"Depth {d}" for d in depths],
                              columns=[m.capitalize().replace('_', ' ') for m in metrics])

    sns.heatmap(df_heatmap, annot=True, cmap='viridis', vmin=0, vmax=1, linewidths=0.5, fmt='.3f')
    plt.title('Detailed XGBoost Performance Comparison')
    plt.tight_layout()
    plt.savefig('charts/XGBoost/skill_recommender/depth_heatmap_comparison.png')
    plt.close()


def test_sample_profiles(base_recommender, model):
    """Test model with sample user profiles"""
    # Ensure SkillRecommender is trained
    try:
        base_recommender.train_model()
    except AttributeError:
        print("SkillRecommender does not have a train_model method. Skipping training.")

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
            recommendations = base_recommender.recommend_skills(profile['skills'], top_n=3)
            for skill, score in recommendations:
                print(f"- {skill} (confidence: {score:.4f})")
        except Exception as e:
            print(f"Error: {e}")


def save_xgboost_model(model, best_depth):
    """Save the trained XGBoost model and related components"""
    # Create models directory if it doesn't exist
    os.makedirs('../models', exist_ok=True)

    # Save the model
    model_path = '../models/course_recommendation_xgboost_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Save the model configuration
    config_path = '../models/course_recommendation_config.pkl'
    config = {
        'max_depth': best_depth,
        'model_type': 'xgboost',
        'created_at': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)

    print(f"\nCourse recommendation XGBoost model saved to {model_path}")
    print(f"Model configuration saved to {config_path}")

    return model_path, config_path


def predict_new_courses(model, recommender, user_skills):
    """Predict recommended courses for a user based on their current skills

    Args:
        model: Trained XGBoost model
        recommender: SkillRecommender instance with feature configuration
        user_skills: List of user's current skills

    Returns:
        List of (skill, confidence) tuples for recommended skills
    """
    try:
        # Convert user skills to feature vector
        if not isinstance(user_skills, list):
            user_skills = [s.strip() for s in user_skills.split(',') if s.strip()]

        # Create one-hot encoding for user skills
        user_vector = np.zeros(len(recommender.feature_names))

        for skill in user_skills:
            # Find skill in feature names (match case-insensitive)
            for i, feature in enumerate(recommender.feature_names):
                if feature.lower() == skill.lower():
                    user_vector[i] = 1
                    break

        # Reshape for prediction
        user_vector = user_vector.reshape(1, -1)

        # Make prediction
        prediction = model.predict_proba(user_vector)

        # Get skill names for each predicted class
        skill_predictions = []

        # Process each output class
        for i, class_probs in enumerate(prediction):
            # Get probability of positive class (class 1)
            if len(class_probs[0]) > 1:  # Binary classifier
                prob = class_probs[0][1]
            else:  # If only one probability is returned
                prob = class_probs[0][0]

            # Get the skill name for this target
            skill_name = recommender.target_names[i]

            # Add to results if probability is significant and skill not already known
            if prob > 0.2 and skill_name.lower() not in [s.lower() for s in user_skills]:
                skill_predictions.append((skill_name, float(prob)))

        # Sort by probability
        skill_predictions.sort(key=lambda x: x[1], reverse=True)

        return skill_predictions

    except Exception as e:
        print(f"Error during course prediction: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    # Ensure output directories exist
    os.makedirs('charts/XGBoost/skill_recommender', exist_ok=True)

    # Run test
    results, model = test_xgboost_recommender()

    # Save results to CSV
    df_results = pd.DataFrame(results).T
    df_results.to_csv('charts/XGBoost/skill_recommender/xgboost_results.csv')

    # Get the best depth
    best_depth = max(results.keys(), key=lambda k: results[k]['f1_micro'])

    # Initialize DataProcessor and recommender for saving base features
    processor = DataProcessor(file_path='../data/edited_skill_exchange_dataset.csv', columns=8)
    processor.df.columns = ["id", "enrollment_date", "current_skills",
                            "desired_skills", "target_skills", "success", "engagement_level", "peer_user_id"]


    # Manual data cleaning instead of using clean_data
    for col in ['current_skills', 'desired_skills', 'target_skills']:
        processor.df[col] = processor.df[col].apply(processor._string_to_list)

    processor.df['success'] = processor.df['success'].astype(bool)

    base_recommender = SkillRecommender(processor)
    base_recommender.prepare_data(use_pca=False)

    # Save the model and related components
    model_path, config_path = save_xgboost_model(model, best_depth)

    # Save feature names
    features_path = '../models/course_recommendation_features.pkl'
    with open(features_path, 'wb') as f:
        pickle.dump({
            'feature_names': base_recommender.feature_names,
            'target_names': base_recommender.target_names
        }, f)
    print(f"Feature information saved to {features_path}")

    # Test inference
    test_skills = ["JavaScript", "HTML", "CSS"]
    print(f"\nTesting inference with skills: {', '.join(test_skills)}")
    recommendations = predict_new_courses(model, base_recommender, test_skills)
    for skill, score in recommendations[:5]:
        print(f"- {skill} (confidence: {score:.4f})")