import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Create directory for charts if it doesn't exist
os.makedirs('./charts', exist_ok=True)

# Load data
print("Loading dataset...")
data = pd.read_csv('../data/edited_skill_exchange_dataset.csv')
df = pd.DataFrame(data)

# Feature engineering
print("Performing feature engineering...")
# Convert joinedDate to datetime and extract features
df['joinedDate'] = pd.to_datetime(df['joinedDate'])
df['days_since_joined'] = (pd.Timestamp('2025-04-24') - df['joinedDate']).dt.days
df['join_month'] = df['joinedDate'].dt.month
df['join_year'] = df['joinedDate'].dt.year

# Convert engagement level to numerical
engagement_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
df['engagement_numeric'] = df['engagement_level'].map(engagement_mapping)

# Convert boolean to numerical
df['isVerified_num'] = df['isVerified'].astype(int)

# Process skills data using MultiLabelBinarizer
def process_skills(series):
    mlb = MultiLabelBinarizer()
    processed = mlb.fit_transform([set(s.split(', ')) for s in series])
    return pd.DataFrame(processed, columns=mlb.classes_)

# Process all skill-related columns
joined_courses_df = process_skills(df['joinedCourses'])
skills_df = process_skills(df['skills'])
desired_skills_df = process_skills(df['desired_skills'])

# Count skills as additional features
df['num_joined_courses'] = df['joinedCourses'].apply(lambda x: len(x.split(', ')))
df['num_skills'] = df['skills'].apply(lambda x: len(x.split(', ')))
df['num_desired_skills'] = df['desired_skills'].apply(lambda x: len(x.split(', ')))

# Create skill overlap features
def calculate_overlap(row):
    user_skills = set(row['skills'].split(', '))
    desired_skills = set(row['desired_skills'].split(', '))
    courses = set(row['joinedCourses'].split(', '))

    return {
        'skills_desired_overlap': len(user_skills.intersection(desired_skills)),
        'skills_course_overlap': len(user_skills.intersection(courses)),
        'desired_course_overlap': len(desired_skills.intersection(courses))
    }

overlaps = df.apply(calculate_overlap, axis=1, result_type='expand')
df = pd.concat([df, overlaps], axis=1)

# Combine all features
X = pd.concat([
    df[['days_since_joined', 'isVerified_num', 'engagement_numeric',
        'join_month', 'join_year', 'num_joined_courses', 'num_skills',
        'num_desired_skills', 'skills_desired_overlap', 'skills_course_overlap',
        'desired_course_overlap']],
    joined_courses_df.add_prefix('course_'),
    skills_df.add_prefix('skill_'),
    desired_skills_df.add_prefix('desired_')
], axis=1)

# Target variable
y = df['peer_user_id']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Train-test split
print("Splitting data and training model...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.25, random_state=42)

# Define SVR model with hyperparameter tuning
param_grid = {
    'C': [10, 100, 1000],
    'gamma': [0.01, 0.1, 1],
    'epsilon': [0.01, 0.1, 1]
}

svr = SVR(kernel='rbf')
grid_search = GridSearchCV(svr, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get best model
best_svr = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# Make predictions
y_pred = best_svr.predict(X_test)
y_pred_train = best_svr.predict(X_train)

# Calculate regression metrics
mse_test = mean_squared_error(y_test, y_pred)
r2_test = r2_score(y_test, y_pred)
mse_train = mean_squared_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

# For classification metrics, define a match threshold
threshold = 1000  # Consider pairs to be a match if predicted within 1000 of actual
y_binary_true = np.ones(len(y_test))  # Assuming all are matches in our data
y_binary_pred = np.abs(y_test - y_pred) < threshold

# Calculate classification metrics
precision = precision_score(y_binary_true, y_binary_pred, zero_division=0)
recall = recall_score(y_binary_true, y_binary_pred, zero_division=0)
f1 = f1_score(y_binary_true, y_binary_pred, zero_division=0)

# For ROC AUC, normalize the prediction errors to be between 0 and 1
max_error = max(np.abs(y_test - y_pred))
scores = 1 - (np.abs(y_test - y_pred) / max_error if max_error > 0 else 0)
auc = roc_auc_score(y_binary_true, scores) if len(np.unique(y_binary_true)) > 1 else 0

# Make predictions for all users
all_predictions = best_svr.predict(X_scaled_df)

# Add predictions to dataframe
df['predicted_peer_user_id'] = all_predictions
df['match_quality'] = np.abs(df['peer_user_id'] - df['predicted_peer_user_id'])
df['good_match'] = df['match_quality'] < threshold

# Display results
print("\nModel Performance Metrics:")
print(f"Test MSE: {mse_test:.2f}")
print(f"Test R² Score: {r2_test:.2f}")
print(f"Train MSE: {mse_train:.2f}")
print(f"Train R² Score: {r2_train:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"AUC-ROC: {auc:.2f}")

# Display matching results
results_df = df[['user_id', 'peer_user_id', 'predicted_peer_user_id', 'match_quality', 'good_match']]
print("\nUser Matching Results:")
print(results_df)

# Visualizations
print("\nGenerating visualizations...")

# 1. Metrics bar chart
plt.figure(figsize=(12, 6))
metrics = ['Test MSE', 'Test R²', 'Train MSE', 'Train R²', 'Precision', 'Recall', 'F1', 'AUC']
values = [mse_test, r2_test, mse_train, r2_train, precision, recall, f1, auc]
colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c', '#34495e', '#95a5a6']

plt.bar(metrics, values, color=colors)
plt.title('SVM Model Performance Metrics', fontsize=15)
plt.ylabel('Score', fontsize=12)
plt.ylim(0, max(max(values) + 0.1, 1.0))
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.grid(axis='y', alpha=0.3)
plt.savefig('./charts/svm_metrics_performance.png')

# 2. Actual vs Predicted scatter plot
plt.figure(figsize=(10, 8))
sns.scatterplot(x=df['peer_user_id'], y=df['predicted_peer_user_id'], alpha=0.7)
plt.plot([min(df['peer_user_id']), max(df['peer_user_id'])],
         [min(df['peer_user_id']), max(df['peer_user_id'])],
         'r--', linewidth=2)
plt.xlabel('Actual Peer User ID', fontsize=12)
plt.ylabel('Predicted Peer User ID', fontsize=12)
plt.title('Actual vs Predicted Peer User IDs (SVM)', fontsize=15)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./charts/svm_actual_vs_predicted.png')

# 3. Feature importance approximation
# Since SVR doesn't provide direct feature importance, use coefficient-based approach
# Train a linear model to get feature coefficients
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Get feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': np.abs(linear_model.coef_)
})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False).head(15)

plt.figure(figsize=(12, 10))
plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='#2980b9')
plt.xlabel('Importance', fontsize=12)
plt.title('Top 15 Feature Importances (SVM)', fontsize=15)
plt.gca().invert_yaxis()  # Invert to show highest values at top
plt.tight_layout()
plt.savefig('./charts/svm_feature_importance.png')

# 4. Prediction error distribution
plt.figure(figsize=(10, 6))
errors = df['match_quality']
sns.histplot(errors, bins=20, kde=True)
plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
plt.xlabel('Prediction Error', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Distribution of Prediction Errors', fontsize=15)
plt.legend()
plt.tight_layout()
plt.savefig('./charts/svm_error_distribution.png')

# 5. Match quality heatmap
plt.figure(figsize=(10, 8))
match_pivot = pd.pivot_table(
    df,
    values='match_quality',
    index='engagement_level',
    columns='isVerified',
    aggfunc='mean'
)
sns.heatmap(match_pivot, annot=True, cmap='YlGnBu', fmt='.1f')
plt.title('Average Match Quality by Engagement and Verification', fontsize=15)
plt.tight_layout()
plt.savefig('./charts/svm_match_quality_heatmap.png')

print("\nAnalysis complete! Charts saved to ./charts/")