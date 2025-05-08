import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os

from xgboost import plot_tree

# Ensure chart directories exist
os.makedirs('charts/xgboost', exist_ok=True)

# Load data
data = pd.read_csv('../data/edited_skill_exchange_dataset.csv')
df = pd.DataFrame(data)

# Data Preprocessing
df['joinedDate'] = pd.to_datetime(df['joinedDate'])
df['days_since_joining'] = (pd.Timestamp.now() - df['joinedDate']).dt.days


# Function to safely split text fields and count items
def count_items(text):
    if pd.isna(text) or not isinstance(text, str):
        return 0
    items = [item.strip() for item in text.split(',') if item.strip()]
    return len(items)


# Count-based features
df['num_joined_courses'] = df['joinedCourses'].apply(count_items)
df['num_skills'] = df['skills'].apply(count_items)
df['num_desired_skills'] = df['desired_skills'].apply(count_items)


# Convert string lists to actual lists for intersection operations
def string_to_list(text):
    if pd.isna(text) or not isinstance(text, str):
        return []
    return [item.strip() for item in text.split(',') if item.strip()]


df['joined_courses_list'] = df['joinedCourses'].apply(string_to_list)
df['skills_list'] = df['skills'].apply(string_to_list)
df['desired_skills_list'] = df['desired_skills'].apply(string_to_list)

# Count overlaps between different skill categories
df['skills_courses_overlap'] = df.apply(
    lambda x: len(set(x['skills_list']).intersection(set(x['joined_courses_list']))), axis=1
)
df['desired_courses_overlap'] = df.apply(
    lambda x: len(set(x['desired_skills_list']).intersection(set(x['joined_courses_list']))), axis=1
)
df['skills_desired_overlap'] = df.apply(
    lambda x: len(set(x['skills_list']).intersection(set(x['desired_skills_list']))), axis=1
)

# Calculate ratios to represent learning progress
df['course_effectiveness'] = df.apply(
    lambda x: x['desired_courses_overlap'] / x['num_joined_courses'] if x['num_joined_courses'] > 0 else 0,
    axis=1
)
df['skills_acquisition_rate'] = df.apply(
    lambda x: x['skills_desired_overlap'] / x['num_desired_skills'] if x['num_desired_skills'] > 0 else 0,
    axis=1
)
df['learning_gap'] = df.apply(
    lambda x: x['num_desired_skills'] - x['skills_desired_overlap'], axis=1
)

# Convert isVerified to numeric
df['isVerified'] = df['isVerified'].map(lambda x: 1 if str(x).lower() == 'true' else 0)

# Features for the model (focusing on counts and numeric relationships)
features = [
    'days_since_joining',
    'num_joined_courses',
    'num_skills',
    'num_desired_skills',
    'skills_courses_overlap',
    'desired_courses_overlap',
    'skills_desired_overlap',
    'course_effectiveness',
    'skills_acquisition_rate',
    'learning_gap',
    'isVerified'
]

# Select features that exist in dataframe
X = df[features].values
y = df['engagement_level']

# Encode the target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42)

# Define hyperparameters manually instead of using GridSearchCV
xgb_params = {
    'max_depth': 5,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'multi:softprob',
    'num_class': len(np.unique(y_encoded)),
    'random_state': 42
}

# Create and train XGBoost model
xgb_model = xgb.XGBClassifier(**xgb_params)
xgb_model.fit(X_train, y_train)

# Evaluate model
y_pred = xgb_model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"RMSE: {rmse:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('XGBoost Confusion Matrix')
plt.tight_layout()
plt.savefig('charts/xgboost/confusion_matrix.png')
plt.close()

# Feature importance from the model
feature_importance = xgb_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importance
})
importance_df = importance_df.sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('XGBoost Feature Importance')
plt.tight_layout()
plt.savefig('charts/xgboost/feature_importance_model.png')
plt.close()

# Learning curve
train_sizes = np.linspace(0.1, 0.99, 5)  # Changed from 1.0 to 0.99
train_scores = []
test_scores = []

for size in train_sizes:
    X_sub_train, _, y_sub_train, _ = train_test_split(X_train, y_train, train_size=size, random_state=42)
    sub_model = xgb.XGBClassifier(**xgb_params)
    sub_model.fit(X_sub_train, y_sub_train)
    train_scores.append(accuracy_score(y_sub_train, sub_model.predict(X_sub_train)))
    test_scores.append(accuracy_score(y_test, sub_model.predict(X_test)))
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores, 'o-', label='Training Score')
plt.plot(train_sizes, test_scores, 'o-', label='Test Score')
plt.xlabel('Training Set Size Fraction')
plt.ylabel('Accuracy Score')
plt.title('XGBoost Learning Curve')
plt.legend()
plt.grid(True)
plt.savefig('charts/xgboost/learning_curve.png')
plt.close()

# Alternative feature importance: train models excluding one feature at a time
feature_importance_alt = []
baseline_score = accuracy_score(y_test, y_pred)

for i, feature in enumerate(features):
    # Create a mask for all columns except the current feature
    mask = np.ones(len(features), dtype=bool)
    mask[i] = False

    # Train a model without this feature
    model_without_feature = xgb.XGBClassifier(**xgb_params)
    model_without_feature.fit(X_train[:, mask], y_train)

    # Predict and calculate accuracy
    y_pred_without = model_without_feature.predict(X_test[:, mask])
    acc_without = accuracy_score(y_test, y_pred_without)

    # The importance is the drop in accuracy when the feature is removed
    importance = baseline_score - acc_without
    feature_importance_alt.append(importance)

# Create a dataframe for the alternative importance
alt_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importance_alt
})
alt_importance_df = alt_importance_df.sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=alt_importance_df)
plt.title('Feature Importance (Drop Column Method)')
plt.tight_layout()
plt.savefig('charts/xgboost/feature_importance_alt.png')
plt.close()
# Visualize a decision tree
"""
plt.figure(figsize=(30, 15))
xgb.plot_tree(xgb_model, num_trees=0, rankdir='LR')
plt.title('XGBoost Decision Tree (Tree 0)')
plt.tight_layout()
plt.savefig('charts/xgboost/decision_tree.png')
plt.close()
"""
"""
# Plot a single tree from the XGBoost model
plt.figure(figsize=(30, 20))  # You can adjust size depending on complexity
plot_tree(xgb_model, num_trees=0, rankdir='LR',  # 'LR' = left to right layout
          fmap='',  # feature map not necessary, already using feature names
          ax=None,
          )
plt.title('Feature-Based Decision Tree (Tree 0 of XGBoost Model)', fontsize=18)
plt.tight_layout()
plt.savefig('charts/xgboost/feature_based_decision_tree.png')
plt.close()
"""

# Create summary report
with open('charts/xgboost/model_summary.txt', 'w') as f:
    f.write("XGBoost Model for User Engagement Prediction\n")
    f.write("==========================================\n\n")

    f.write("1. Data Preprocessing Steps:\n")
    f.write("   - Converted joinedDate to datetime\n")
    f.write("   - Created days_since_joining feature\n")
    f.write("   - Counted items in joinedCourses, skills, and desired_skills\n")
    f.write("   - Calculated overlaps between different skill categories\n")
    f.write("   - Added derived features for learning progress and effectiveness\n\n")

    f.write("2. Features Used:\n")
    for feature in features:
        f.write(f"   - {feature}\n")
    f.write("\n")

    f.write("3. Model Parameters:\n")
    for param, value in xgb_params.items():
        f.write(f"   - {param}: {value}\n")
    f.write("\n")

    f.write("4. Evaluation Metrics:\n")
    f.write(f"   - Accuracy: {accuracy:.4f}\n")
    f.write(f"   - Precision: {precision:.4f}\n")
    f.write(f"   - Recall: {recall:.4f}\n")
    f.write(f"   - F1 Score: {f1:.4f}\n")
    f.write(f"   - RMSE: {rmse:.4f}\n\n")

    f.write("5. Top 5 Most Important Features (Model-based):\n")
    top_features = importance_df.head(5)
    for idx, row in top_features.iterrows():
        f.write(f"   - {row['Feature']}: {row['Importance']:.4f}\n")
    f.write("\n")

    f.write("6. Top 5 Most Important Features (Alternative Method):\n")
    top_alt_features = alt_importance_df.head(5)
    for idx, row in top_alt_features.iterrows():
        f.write(f"   - {row['Feature']}: {row['Importance']:.4f}\n")


# Add at the end of EngagementScore/XGBOOST.py
import pickle
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save the model
model_path = '../models/engagement_xgboost_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(xgb_model, f)

# Save the label encoder
le_path = '../models/engagement_label_encoder.pkl'
with open(le_path, 'wb') as f:
    pickle.dump(le, f)

# Save feature list
features_path = '../models/engagement_features.pkl'
with open(features_path, 'wb') as f:
    pickle.dump(features, f)

print(f"\nEngagement XGBoost model saved to {model_path}")
print(f"Label encoder saved to {le_path}")
print(f"Feature list saved to {features_path}")