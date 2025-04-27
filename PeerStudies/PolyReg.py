import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')

# Create charts directory if it doesn't exist
os.makedirs('./charts', exist_ok=True)

# Create DataFrame from the provided data
data = pd.read_csv('../data/edited_skill_exchange_dataset.csv')

df = pd.DataFrame(data)

# Convert joinedDate to datetime and extract features
df['joinedDate'] = pd.to_datetime(df['joinedDate'])
df['days_since_joined'] = (pd.Timestamp('2025-04-24') - df['joinedDate']).dt.days

# Convert engagement_level to numerical
engagement_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
df['engagement_level_num'] = df['engagement_level'].map(engagement_mapping)

# Convert boolean to numerical
df['isVerified_num'] = df['isVerified'].astype(int)


# Process skills data using MultiLabelBinarizer
def process_skills(series):
    mlb = MultiLabelBinarizer()
    processed = mlb.fit_transform([set(s.split(', ')) for s in series])
    return pd.DataFrame(processed, columns=mlb.classes_), mlb.classes_


# Process all skill-related columns
joined_courses_df, joined_courses_cols = process_skills(df['joinedCourses'])
skills_df, skills_cols = process_skills(df['skills'])
desired_skills_df, desired_skills_cols = process_skills(df['desired_skills'])

# Combine all features
X = pd.concat([
    df[['days_since_joined', 'isVerified_num', 'engagement_level_num']],
    joined_courses_df,
    skills_df,
    desired_skills_df
], axis=1)

y = df['peer_user_id']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# Function to evaluate models of different polynomial degrees
def evaluate_polynomial_models(max_degree=3):
    results = []
    for degree in range(1, max_degree + 1):
        # Create polynomial regression model
        poly_model = make_pipeline(
            PolynomialFeatures(degree=degree, include_bias=False),
            LinearRegression()
        )

        # Train the model
        poly_model.fit(X_train, y_train)

        # Make predictions
        y_pred_train = poly_model.predict(X_train)
        y_pred_test = poly_model.predict(X_test)

        # Calculate metrics
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)

        # For classification metrics, convert to binary matching outcome
        def calculate_binary_metrics(y_true, y_pred):
            # Convert to binary (1 = good match, 0 = bad match)
            y_binary_true = np.ones(len(y_true))  # All are actual matches in our data
            y_binary_pred = np.abs(y_pred - y_true) < 1000

            # Calculate metrics
            precision = precision_score(y_binary_true, y_binary_pred, zero_division=0)
            recall = recall_score(y_binary_true, y_binary_pred, zero_division=0)
            f1 = f1_score(y_binary_true, y_binary_pred, zero_division=0)

            # For ROC AUC, we need probability-like scores
            max_error = max(np.abs(y_pred - y_true))
            scores = 1 - (np.abs(y_pred - y_true) / max_error if max_error > 0 else 0)

            try:
                auc = roc_auc_score(y_binary_true, scores)
            except:
                auc = None

            return precision, recall, f1, auc

        test_precision, test_recall, test_f1, test_auc = calculate_binary_metrics(y_test, y_pred_test)

        results.append({
            'Degree': degree,
            'MSE_Train': mse_train,
            'MSE_Test': mse_test,
            'R2_Train': r2_train,
            'R2_Test': r2_test,
            'Precision': test_precision,
            'Recall': test_recall,
            'F1': test_f1,
            'AUC': test_auc if test_auc else 0,
            'Model': poly_model,
            'y_pred_test': y_pred_test
        })

    return results


# Evaluate models of different polynomial degrees
model_results = evaluate_polynomial_models(max_degree=3)

# Create visualizations for model comparison
plt.figure(figsize=(12, 8))
degrees = [result['Degree'] for result in model_results]
mse_train = [result['MSE_Train'] for result in model_results]
mse_test = [result['MSE_Test'] for result in model_results]

bar_width = 0.35
index = np.arange(len(degrees))

plt.bar(index, mse_train, bar_width, label='Training MSE')
plt.bar(index + bar_width, mse_test, bar_width, label='Testing MSE')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.title('MSE by Polynomial Degree')
plt.xticks(index + bar_width / 2, degrees)
plt.legend()
plt.savefig('./charts/poly_mse_comparison.png')

# R2 Score comparison
plt.figure(figsize=(12, 8))
r2_train = [result['R2_Train'] for result in model_results]
r2_test = [result['R2_Test'] for result in model_results]

plt.bar(index, r2_train, bar_width, label='Training R²')
plt.bar(index + bar_width, r2_test, bar_width, label='Testing R²')
plt.xlabel('Polynomial Degree')
plt.ylabel('R² Score')
plt.title('R² Score by Polynomial Degree')
plt.xticks(index + bar_width / 2, degrees)
plt.legend()
plt.savefig('./charts/poly_r2_comparison.png')

# Classification metrics comparison
plt.figure(figsize=(12, 8))
metrics = ['Precision', 'Recall', 'F1', 'AUC']
for i, metric in enumerate(metrics):
    plt.subplot(2, 2, i + 1)
    values = [result[metric] for result in model_results]
    plt.bar(degrees, values)
    plt.xlabel('Polynomial Degree')
    plt.ylabel(metric)
    plt.title(f'{metric} by Polynomial Degree')
plt.tight_layout()
plt.savefig('./charts/poly_classification_metrics.png')

# Select the best model based on test MSE
best_model_idx = np.argmin([result['MSE_Test'] for result in model_results])
best_model = model_results[best_model_idx]['Model']
best_degree = model_results[best_model_idx]['Degree']

# Generate predictions using the best model
df['predicted_peer_user_id'] = best_model.predict(X)
df['match_quality'] = np.abs(df['peer_user_id'] - df['predicted_peer_user_id'])
df['good_match'] = df['match_quality'] < 1000

# Create actual vs predicted plot for the best model
best_y_pred = model_results[best_model_idx]['y_pred_test']
plt.figure(figsize=(10, 6))
plt.scatter(y_test, best_y_pred)
plt.xlabel('Actual peer_user_id')
plt.ylabel('Predicted peer_user_id')
plt.title(f'Actual vs Predicted peer_user_id (Degree {best_degree})')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.savefig('./charts/poly_actual_vs_predicted.png')

# Create residual plot for the best model
residuals = y_test - best_y_pred
plt.figure(figsize=(10, 6))
plt.scatter(best_y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Predicted peer_user_id')
plt.ylabel('Residuals')
plt.title(f'Residual Plot (Degree {best_degree})')
plt.savefig('./charts/poly_residuals.png')

# Print comparison of all models
comparison_df = pd.DataFrame({
    'Degree': [result['Degree'] for result in model_results],
    'Training MSE': [result['MSE_Train'] for result in model_results],
    'Testing MSE': [result['MSE_Test'] for result in model_results],
    'Training R²': [result['R2_Train'] for result in model_results],
    'Testing R²': [result['R2_Test'] for result in model_results],
    'Precision': [result['Precision'] for result in model_results],
    'Recall': [result['Recall'] for result in model_results],
    'F1 Score': [result['F1'] for result in model_results],
    'AUC-ROC': [result['AUC'] for result in model_results]
})

print("Model Comparison:")
print(comparison_df)

# Print the best model results
print(f"\nBest Model: Polynomial Regression (Degree {best_degree})")
print(f"Test MSE: {model_results[best_model_idx]['MSE_Test']:.2f}")
print(f"Test R²: {model_results[best_model_idx]['R2_Test']:.2f}")
print(f"Precision: {model_results[best_model_idx]['Precision']:.2f}")
print(f"Recall: {model_results[best_model_idx]['Recall']:.2f}")
print(f"F1 Score: {model_results[best_model_idx]['F1']:.2f}")
print(
    f"AUC-ROC: {model_results[best_model_idx]['AUC']:.2f}" if model_results[best_model_idx]['AUC'] else "AUC-ROC: N/A")

# Print results for all users
results_df = df[['user_id', 'peer_user_id', 'predicted_peer_user_id', 'match_quality', 'good_match']]
print("\nMatching Results:")
print(results_df)

print("\nCharts saved to ./charts/")