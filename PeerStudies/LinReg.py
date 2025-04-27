import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')

# Create charts directory if it doesn't exist
os.makedirs('./charts', exist_ok=True)

# load data from CSV
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

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print metrics
print("Mean Squared Error:", mse)
print("R² Score:", r2)

# Feature importance analysis
feature_names = list(X.columns)
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': model.coef_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)


# For classification metrics, we need to convert predictions to binary classes
# Let's create a simple metric: if the difference between prediction and actual is less than 1000, consider it a match
# This is a simplified approach for demonstration
def calculate_binary_metrics(y_true, y_pred):
    # Convert to binary (1 = good match, 0 = bad match)
    y_binary_true = np.ones(len(y_true))  # All are actual matches in our data
    y_binary_pred = np.abs(y_pred - y_true) < 1000

    # Calculate metrics
    precision = precision_score(y_binary_true, y_binary_pred, zero_division=0)
    recall = recall_score(y_binary_true, y_binary_pred, zero_division=0)
    f1 = f1_score(y_binary_true, y_binary_pred, zero_division=0)

    # For ROC AUC, we need probability-like scores
    # Normalize the errors to be between 0 and 1 (higher is better match)
    max_error = max(np.abs(y_pred - y_true))
    scores = 1 - (np.abs(y_pred - y_true) / max_error if max_error > 0 else 0)

    try:
        auc = roc_auc_score(y_binary_true, scores)
    except:
        auc = None

    return precision, recall, f1, auc


# Calculate binary metrics for test data
test_precision, test_recall, test_f1, test_auc = calculate_binary_metrics(y_test, y_pred)

# Create visualizations
plt.figure(figsize=(10, 6))
plt.bar(feature_importance['Feature'][:10], feature_importance['Importance'][:10])
plt.xticks(rotation=90)
plt.title('Top 10 Feature Importance')
plt.tight_layout()
plt.savefig('./charts/feature_importance.png')

# Create actual vs predicted plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual peer_user_id')
plt.ylabel('Predicted peer_user_id')
plt.title('Actual vs Predicted peer_user_id')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.savefig('./charts/actual_vs_predicted.png')

# Create metrics bar chart
plt.figure(figsize=(10, 6))
metrics = ['MSE', 'R²', 'Precision', 'Recall', 'F1', 'AUC']
values = [mse, r2, test_precision, test_recall, test_f1, test_auc if test_auc else 0]
plt.bar(metrics, values)
plt.title('Model Performance Metrics')
plt.ylim(0, max(values) * 1.1)
plt.savefig('./charts/metrics.png')

# Create residual plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Predicted peer_user_id')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.savefig('./charts/residuals.png')

# Create a table with results for all users
df['predicted_peer_user_id'] = model.predict(X)
df['match_quality'] = np.abs(df['peer_user_id'] - df['predicted_peer_user_id'])
df['good_match'] = df['match_quality'] < 1000

# Display results
results_df = df[['user_id', 'peer_user_id', 'predicted_peer_user_id', 'match_quality', 'good_match']]
print("\nMatching Results:")
print(results_df)

# Print overall model performance
print("\nModel Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")
print(f"Precision: {test_precision:.2f}")
print(f"Recall: {test_recall:.2f}")
print(f"F1 Score: {test_f1:.2f}")
print(f"AUC-ROC: {test_auc:.2f}" if test_auc else "AUC-ROC: N/A")

print("\nCharts saved to ./charts/")