import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Ensure chart directories exist
os.makedirs('charts/knn', exist_ok=True)

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
X = df[features]
y = df['engagement_level']

# Encode the target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42)

# Create pipeline with scaling and KNN
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

# Grid search for hyperparameter tuning
param_grid = {
    'knn__n_neighbors': [3, 5, 7, 9, 11],
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan']
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1_weighted')
grid_search.fit(X_train, y_train)

best_pipeline = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# Evaluate model
y_pred = best_pipeline.predict(X_test)

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
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('charts/knn/confusion_matrix.png')
plt.close()

# Feature importance using permutation method
feature_importance = []
base_accuracy = accuracy_score(y_test, y_pred)

for i, feature in enumerate(features):
    # Make a copy of the test data
    X_test_permuted = X_test.copy()
    # Shuffle values for current feature
    X_test_permuted[feature] = np.random.permutation(X_test_permuted[feature].values)
    # Predict with shuffled feature
    y_permuted_pred = best_pipeline.predict(X_test_permuted)
    # Calculate accuracy drop
    accuracy_drop = base_accuracy - accuracy_score(y_test, y_permuted_pred)
    feature_importance.append(accuracy_drop)

# Visualize feature importance
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importance
})
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance (Impact on Model Accuracy)')
plt.tight_layout()
plt.savefig('charts/knn/feature_importance.png')
plt.close()

# Find optimal K value
k_values = list(range(1, 21, 2))
accuracy_scores = []

for k in k_values:
    temp_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=k))
    ])
    temp_pipeline.fit(X_train, y_train)
    y_pred_k = temp_pipeline.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred_k))

plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracy_scores, marker='o')
plt.title('Accuracy vs. k Value')
plt.xlabel('k Value')
plt.ylabel('Accuracy')
plt.grid(True)
plt.savefig('charts/knn/accuracy_vs_k.png')
plt.close()

# Create summary report
with open('charts/knn/model_summary.txt', 'w') as f:
    f.write("KNN Model for User Engagement Prediction\n")
    f.write("=======================================\n\n")

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

    f.write("3. Best Model Parameters:\n")
    for param, value in grid_search.best_params_.items():
        f.write(f"   - {param}: {value}\n")
    f.write("\n")

    f.write("4. Evaluation Metrics:\n")
    f.write(f"   - Accuracy: {accuracy:.4f}\n")
    f.write(f"   - Precision: {precision:.4f}\n")
    f.write(f"   - Recall: {recall:.4f}\n")
    f.write(f"   - F1 Score: {f1:.4f}\n")
    f.write(f"   - RMSE: {rmse:.4f}\n\n")

    f.write("5. Top 5 Most Important Features:\n")
    top_features = feature_importance_df.head(5)
    for idx, row in top_features.iterrows():
        f.write(f"   - {row['Feature']}: {row['Importance']:.4f}\n")