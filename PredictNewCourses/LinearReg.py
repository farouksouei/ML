import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import time

# Add parent directory to path for imports
sys.path.append('..')

from DataProcessor import DataProcessor
from PredictNewCourses.KNN import SkillRecommender


def test_linear_models():
    """Test and evaluate Linear Regression, Lasso, and Ridge models."""
    warnings.filterwarnings("ignore", category=UserWarning)
    print("=== Testing Linear Models for Skill Recommendation ===")

    # Initialize DataProcessor and load data
    processor = DataProcessor(file_path='../data/edited_skill_exchange_dataset.csv', columns=6)
    processor.df.columns = ["id", "enrollment_date", "current_skills", "desired_skills", "target_skills", "success"]
    processor.clean_data()
    print(f"Loaded dataset with {len(processor.df)} entries")

    # Initialize recommender for data preparation
    base_recommender = SkillRecommender(processor)
    base_recommender.prepare_data(use_pca=False)  # Use raw features for interpretability

    results = {}
    models = {}

    # Run experiments for Linear Regression, Lasso, and Ridge
    for model_name, model_type, params in [
        ("Linear Regression", LinearRegression, {}),
        ("Lasso", Lasso, {"alpha": [0.1, 1.0, 10.0]}),
        ("Ridge", Ridge, {"alpha": [0.1, 1.0, 10.0]})
    ]:
        print(f"\n--- Testing {model_name} ---")
        training_start = time.time()

        # Train models (one for each target skill)
        if model_name == "Linear Regression":
            trained_models = []
            for i in range(base_recommender.y_train.shape[1]):
                model = model_type()
                model.fit(base_recommender.X_train, base_recommender.y_train[:, i])
                trained_models.append(model)
        else:
            # Use GridSearchCV for Lasso and Ridge to tune alpha
            grid_search = GridSearchCV(
                estimator=model_type(),
                param_grid=params,
                scoring='neg_mean_squared_error',
                cv=3
            )
            trained_models = []
            for i in range(base_recommender.y_train.shape[1]):
                grid_search.fit(base_recommender.X_train, base_recommender.y_train[:, i])
                best_model = grid_search.best_estimator_
                trained_models.append(best_model)

        training_time = time.time() - training_start
        print(f"Training time: {training_time:.2f} seconds")

        # Evaluate models
        pred_start = time.time()
        y_pred_prob = np.zeros(base_recommender.y_test.shape)
        for i, model in enumerate(trained_models):
            y_pred_prob[:, i] = model.predict(base_recommender.X_test)

        # Convert probabilities to binary predictions using 0.5 threshold
        y_pred = (y_pred_prob >= 0.5).astype(int)

        prediction_time = time.time() - pred_start
        print(f"Prediction time: {prediction_time:.2f} seconds")

        # Calculate metrics
        mse = mean_squared_error(base_recommender.y_test, y_pred_prob)
        r2 = r2_score(base_recommender.y_test, y_pred_prob)
        accuracy = (base_recommender.y_test == y_pred).mean()
        precision = precision_score(base_recommender.y_test, y_pred, average='micro', zero_division=0)
        recall = recall_score(base_recommender.y_test, y_pred, average='micro', zero_division=0)
        f1_micro = f1_score(base_recommender.y_test, y_pred, average='micro', zero_division=0)
        f1_macro = f1_score(base_recommender.y_test, y_pred, average='macro', zero_division=0)

        results[model_name] = {
            'mse': mse,
            'r2': r2,
            'accuracy': accuracy,
            'precision_micro': precision,
            'recall_micro': recall,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'training_time': training_time,
            'prediction_time': prediction_time
        }

        models[model_name] = trained_models

        # Display metrics
        print(f"\n=== {model_name} Performance Metrics ===")
        print(f"MSE: {mse:.4f}")
        print(f"R^2: {r2:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision (Micro): {precision:.4f}")
        print(f"Recall (Micro): {recall:.4f}")
        print(f"F1 (Micro): {f1_micro:.4f}")
        print(f"F1 (Macro): {f1_macro:.4f}")

        # Save results
        save_results(results, model_name)

    return results, models, base_recommender


def save_results(results, model_name):
    """Save results to a CSV file."""
    os.makedirs(f'charts/{model_name.replace(" ", "")}/skill_recommender', exist_ok=True)
    df_results = pd.DataFrame([results[model_name]])
    df_results.to_csv(f'charts/{model_name.replace(" ", "")}/skill_recommender/{model_name.lower().replace(" ", "_")}_results.csv')


def visualize_coefficients(recommender, models, model_name):
    """Visualize the most influential coefficients."""
    if recommender.pca is None:
        os.makedirs(f'charts/{model_name.replace(" ", "")}/skill_recommender', exist_ok=True)

        all_coefs = np.zeros(recommender.X_train.shape[1])
        for model in models:
            if hasattr(model, 'coef_'):
                all_coefs += np.abs(model.coef_)

        avg_coefs = all_coefs / len(models)

        # Get top 15 feature coefficients
        top_indices = np.argsort(avg_coefs)[-15:]
        top_coefs = avg_coefs[top_indices]
        top_features = np.array(recommender.feature_names)[top_indices]

        # Plot coefficients
        plt.figure(figsize=(12, 10))
        y_pos = np.arange(len(top_indices))

        plt.barh(y_pos, top_coefs, align='center')
        plt.yticks(y_pos, top_features)
        plt.xlabel('Average Absolute Coefficient Value')
        plt.title(f'Top 15 Most Influential Features in {model_name}')
        plt.tight_layout()
        plt.savefig(f'charts/{model_name.replace(" ", "")}/skill_recommender/feature_coefficients.png')
        plt.close()


if __name__ == "__main__":
    # Run test
    results, models, base_recommender = test_linear_models()

    # Visualize coefficients for each model
    for model_name, trained_models in models.items():
        visualize_coefficients(base_recommender, trained_models, model_name)

    print("\n=== Linear Models Testing Complete ===")