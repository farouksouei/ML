import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import os

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
# Process skills data using MultiLabelBinarizer
def process_skills(series):
    mlb = MultiLabelBinarizer()
    # Handle potential missing values or non-string inputs
    clean_series = series.fillna('').astype(str)
    processed = mlb.fit_transform([set(s.split(', ')) if s else set() for s in clean_series])
    return pd.DataFrame(processed, columns=mlb.classes_)

# Process all skill-related columns
joined_courses_df = process_skills(df['joinedCourses'])
skills_df = process_skills(df['skills'])
desired_skills_df = process_skills(df['desired_skills'])

# Combine all features
X = pd.concat([
    df[['days_since_joined', 'isVerified_num', 'engagement_level_num']],
    joined_courses_df
], axis=1)

# Use desired_skills as target instead of peer_user_id (which seems more appropriate)
# Because peer_user_id is likely an identifier, not a prediction target
y = desired_skills_df

# Split data into training and testing sets - split indices, not the data itself
indices = range(len(X))
train_idx, test_idx = train_test_split(indices, test_size=0.25, random_state=42)

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# Create and train an XGBoost model for multilabel classification
# We'll use XGBClassifier instead of XGBRegressor since we're predicting skills
model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)
model.fit(X_train, y_train)

# Process all skill-related columns
joined_courses_df = process_skills(df['joinedCourses'])
skills_df = process_skills(df['skills'])
desired_skills_df = process_skills(df['desired_skills'])

# Combine all features
X = pd.concat([
    df[['days_since_joined', 'isVerified_num', 'engagement_level_num']],
    joined_courses_df,
    skills_df
], axis=1)




y = df['peer_user_id']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create and train a simple XGBoost model
model = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1)
model.fit(X_train, y_train)

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Calculate metrics
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

# Print metrics
print("\nXGBoost Model Performance:")
print(f"Training MSE: {mse_train:.2f}")
print(f"Testing MSE: {mse_test:.2f}")
print(f"Training R²: {r2_train:.2f}")
print(f"Testing R²: {r2_test:.2f}")

# Feature importance
plt.figure(figsize=(10, 6))
plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
plt.title('XGBoost Feature Importance')
plt.savefig('./charts/xgb_feature_importance.png')

# Create actual vs predicted plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test)
plt.xlabel('Actual peer_user_id')
plt.ylabel('Predicted peer_user_id')
plt.title('XGBoost: Actual vs Predicted peer_user_id')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.savefig('./charts/xgb_actual_vs_predicted.png')

# Create metrics bar chart
plt.figure(figsize=(10, 6))
metrics = ['Training MSE', 'Testing MSE', 'Training R²', 'Testing R²']
values = [mse_train, mse_test, r2_train, r2_test]
plt.bar(metrics, values)
plt.title('XGBoost Model Performance Metrics')
plt.savefig('./charts/xgb_metrics.png')

# Generate predictions for all users
df['predicted_peer_user_id'] = model.predict(X)
df['match_quality'] = np.abs(df['peer_user_id'] - df['predicted_peer_user_id'])
df['good_match'] = df['match_quality'] < 1000

# Show matching results
results_df = df[['user_id', 'peer_user_id', 'predicted_peer_user_id', 'match_quality', 'good_match']]
print("\nUser Matching Results:")
print(results_df)

print("\nCharts saved to ./charts/")